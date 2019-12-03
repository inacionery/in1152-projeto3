import json
import numpy
import pandas
import requests
from scipy import stats
from sklearn import preprocessing,tree
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

originalDataFrame = pandas.read_table('dataset_TSMC2014_NYC.txt',sep='\t')
originalDataFrame.columns = ['UserId','VenueId','VenueCategoryId','VenueCategoryName','Latitude','Longitude','Offset','UtcTime']
originalDataFrame = originalDataFrame.drop(['Offset', 'UtcTime'], axis=1)

categoryDataFrame = pandas.read_table('category_subcategory.csv',sep=';')
categoryDataFrame.columns = ['PrincipalVenueCategoryName','VenueCategoryName','VenueCategoryId']
categoryDataFrame = categoryDataFrame.drop('VenueCategoryName', axis=1)

processedDataFrame = pandas.merge(originalDataFrame, categoryDataFrame, on='VenueCategoryId')

def createCluster():
    groupedDataFrame = processedDataFrame.groupby(['PrincipalVenueCategoryName','UserId']).size().reset_index(name='CheckIns').pivot_table(index='UserId', columns='PrincipalVenueCategoryName', fill_value=0, values='CheckIns').reset_index()

    rowCount,colCount = groupedDataFrame.shape

    for i in range(1,colCount):
        for j in range(0,rowCount):
            average = stats.trim_mean(groupedDataFrame.iloc[:,i],0.1)
            if (groupedDataFrame.iloc[j,i] > average):
                groupedDataFrame.iloc[j,i] = 1
            else:
                groupedDataFrame.iloc[j,i] = 0

    pointCut = int(rowCount * 0.9)

    trainingDataFrame = groupedDataFrame.iloc[:pointCut,:]
    testDataFrame = groupedDataFrame.iloc[pointCut:,:]

    kmeansResult = KMeans(n_clusters = 7,n_init=1).fit(preprocessing.scale(trainingDataFrame.values))

    trainingDataFrame = trainingDataFrame.assign(KMeans = kmeansResult.labels_)

    columns = trainingDataFrame.columns[1:10]

    classifier = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2, min_samples_leaf=1)

    classifier.fit(trainingDataFrame[columns], trainingDataFrame['KMeans'])

    predictClasses = classifier.predict(testDataFrame[columns])

    return groupedDataFrame, trainingDataFrame, predictClasses

def getCategoriesDataFrame():
    categoriesDataFrame = categoryDataFrame['PrincipalVenueCategoryName'].value_counts().reset_index()

    categoriesDataFrame.columns = ['PrincipalVenueCategoryName','CheckIns']

    categoriesDataFrame = categoriesDataFrame.sort_values(['CheckIns'],ascending = [0])

    categoriesDataFrame = categoriesDataFrame['PrincipalVenueCategoryName']

    return categoriesDataFrame

def getUnvisitedVenuesDataFrame(userId, userSimilarityDataFrame, venueFrequencyDataFrame, category):
    testDataFrame = processedDataFrame.loc[(processedDataFrame['UserId']==userId) & (processedDataFrame['PrincipalVenueCategoryName']==category)]
    testDataFrame = testDataFrame['VenueId']
    testDataFrame = list(set(testDataFrame))

    similarUserVenuesDataFrame = pandas.DataFrame()

    for i in range(0,4):
        similarUserVenuesNew = processedDataFrame.loc[(processedDataFrame['UserId']==userSimilarityDataFrame.iloc[5*(userId-1)+i,0]) & (processedDataFrame['PrincipalVenueCategoryName']==category)]
        similarUserVenuesNew = similarUserVenuesNew['VenueId']
        similarUserVenuesDataFrame = pandas.concat([similarUserVenuesDataFrame,pandas.DataFrame(similarUserVenuesNew)])

    similarUserVenuesDataFrame.columns = ['similarUserVenuesNew']
    similarUserVenuesDataFrame = similarUserVenuesDataFrame.drop_duplicates()

    unvisitedVenuesDataFrame = similarUserVenuesDataFrame.loc[~(similarUserVenuesDataFrame['similarUserVenuesNew'].isin(testDataFrame))]
    unvisitedVenuesDataFrame = venueFrequencyDataFrame.loc[venueFrequencyDataFrame['VenueId'].isin(unvisitedVenuesDataFrame['similarUserVenuesNew'])]

    if unvisitedVenuesDataFrame.shape[0]<5:
        venueCategoryList = list(set(processedDataFrame.loc[processedDataFrame['PrincipalVenueCategoryName']==category]['VenueId']))
        venueCategoryList = venueFrequencyDataFrame.loc[venueFrequencyDataFrame['VenueId'].isin(venueCategoryList)]

        venueCategoryList = venueCategoryList[~(venueCategoryList['VenueId'].isin(testDataFrame))]
        venueCategoryList = venueCategoryList[~(venueCategoryList['VenueId'].isin(venueFrequencyDataFrame['VenueId']))]

        x = 5-(unvisitedVenuesDataFrame.shape[0])

        unvisitedVenuesDataFrame = pandas.concat([unvisitedVenuesDataFrame,venueCategoryList.iloc[0:x,]])

    return unvisitedVenuesDataFrame

def	getUserSimilarityDataFrame(groupedDataFrame, trainingDataFrame, predictClasses):
    groupedSubCategoryDataFrame = processedDataFrame.groupby(['VenueCategoryName','UserId']).size().reset_index(name='CheckIns').pivot_table(index='UserId', columns='VenueCategoryName', fill_value=0, values='CheckIns').reset_index()

    pointCut = int(groupedSubCategoryDataFrame.shape[0] * 0.9)

    testDataFrame = groupedSubCategoryDataFrame.iloc[pointCut:,:]
    groupedSubCategoryDataFrame = groupedSubCategoryDataFrame.iloc[:pointCut,:]

    userSimilarityDataFrame = pandas.DataFrame()

    for i in range(0,testDataFrame.shape[0]):
        groupedSubDataFrame = groupedDataFrame.iloc[:,0][trainingDataFrame.loc[trainingDataFrame['KMeans'] == predictClasses[i]].index]
        groupedSubDataFrame.columns = ['UserId','KMeans']
        groupedSubDataFrame = pandas.merge(groupedSubCategoryDataFrame, groupedSubDataFrame, on=['UserId']).reset_index()
        groupedSubDataFrame = groupedSubDataFrame[groupedSubCategoryDataFrame.columns]

        userSimilarity = [None]*groupedSubDataFrame.shape[0]

        for j in range(0,groupedSubDataFrame.shape[0]):
            x= numpy.array(list(groupedSubDataFrame.iloc[j,1:].values)).reshape(1,-1)
            y = numpy.array(list(testDataFrame.iloc[i,1:].values)).reshape(1,-1)
            try :
                userSimilarity[j] = cosine_similarity(x,y)[0][0]
            except:
                userSimilarity[j] = 0

        resultDataFrame = pandas.DataFrame([list(groupedSubDataFrame.iloc[:,0]),userSimilarity]).T

        resultDataFrame.columns = ['SimilarUser','Similarity']
        resultDataFrame = resultDataFrame.sort_values(['Similarity'],ascending=[0])
        resultDataFrame = resultDataFrame.iloc[0:5,:]
        resultDataFrame = resultDataFrame.assign(User = i + 1)

        userSimilarityDataFrame = pandas.concat([userSimilarityDataFrame, resultDataFrame])

    return userSimilarityDataFrame

def getVenueFrequencyDataFrame():
    venueFrequencyDataFrame = processedDataFrame['VenueId'].value_counts().reset_index()

    venueFrequencyDataFrame.columns = ['VenueId','CheckIns']

    venueFrequencyDataFrame = venueFrequencyDataFrame.sort_values(['CheckIns'],ascending = [0])

    return venueFrequencyDataFrame

def main():
    venueFrequencyDataFrame = getVenueFrequencyDataFrame()
    groupedDataFrame, trainingDataFrame, predictClasses = createCluster()
    userSimilarityDataFrame = getUserSimilarityDataFrame(groupedDataFrame, trainingDataFrame, predictClasses)
    categoriesDataFrame = getCategoriesDataFrame()

    again = 'y'
    while again!='n':
        userId = int(input('User ID: - \n'))
        print('Category Number \n')
        print(categoriesDataFrame.to_string())
        category = int(input('\nCategory Number:'))

        category = categoriesDataFrame.iloc[category]

        unvisitedVenuesDataFrame = getUnvisitedVenuesDataFrame(userId, userSimilarityDataFrame, venueFrequencyDataFrame, category)

        unvisitedVenuesDataFrame = pandas.merge(unvisitedVenuesDataFrame, processedDataFrame, on=['VenueId']).drop(['UserId', 'VenueCategoryId', 'PrincipalVenueCategoryName'], axis=1).drop_duplicates(subset ="VenueId").sort_values(['CheckIns'],ascending = [0]).drop(['CheckIns'], axis=1)

        if  unvisitedVenuesDataFrame.shape[0]>0:
            print('Top recommendations:\n')
            pandas.set_option('display.max_rows', None)
            pandas.set_option('display.max_columns', None)
            pandas.set_option('display.width', None)
            pandas.set_option('display.max_colwidth', -1)
            print(unvisitedVenuesDataFrame.to_string(index=False))
        else:
            print('No Recommendations Available')

        again = input(('\nDo you wish to continue (y/n): '))

main()
