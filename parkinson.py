import pandas as pd

data = pd.read_csv('parkinson.csv')
'''
PatientID                     int64
Age                           int64
Gender                        int64
Ethnicity                     int64
EducationLevel                int64
BMI                         float64
Smoking                       int64
AlcoholConsumption          float64
PhysicalActivity            float64
DietQuality                 float64
SleepQuality                float64
FamilyHistoryParkinsons       int64
TraumaticBrainInjury          int64
Hypertension                  int64
Diabetes                      int64
Depression                    int64
Stroke                        int64
SystolicBP                    int64
DiastolicBP                   int64
CholesterolTotal            float64
CholesterolLDL              float64
CholesterolHDL              float64
CholesterolTriglycerides    float64
UPDRS                       float64
MoCA                        float64
FunctionalAssessment        float64
Tremor                        int64
Rigidity                      int64
Bradykinesia                  int64
PosturalInstability           int64
SpeechProblems                int64
SleepDisorders                int64
Constipation                  int64
Diagnosis                     int64
DoctorInCharge               object
'''

# drop DoctorInCharge & PatientID
target_ = data.Diagnosis
data.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'], inplace=True)

# splitting data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target_, shuffle=True, train_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
def average_score(data, target_):
    # average score of model
    average_model = []
    for i in range(3):
        xtrain, xtest, ytrain, ytest = train_test_split(data, target_, shuffle=True, train_size=0.2)
        model.fit(data, target_)
        average_model.append(model.score(xtest, ytest))
    print(average_model)

'''average_score(data, target_)''' # [0.8206650831353919, 0.8123515439429929, 0.8218527315914489]

from sklearn.cluster import KMeans
# Kmeans Clustering
def Kmeans(xtrain):
    kmeans = KMeans(n_clusters = 10, random_state = 0, n_init='auto')
    kmeans.fit(xtrain)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(data = xtrain, x = 'AlcoholConsumption', y = 'Age', hue = kmeans.labels_)
    plt.show()
'''Kmeans(xtrain)'''