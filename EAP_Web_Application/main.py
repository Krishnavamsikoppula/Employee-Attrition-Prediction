import os
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
import pickle
from sklearn.naive_bayes import GaussianNB

from flask import Flask,request,render_template,session
import pandas as pd

from model_testing import nb

train_data = None
test_data = None
data_rows = None
data_columns = None
test_data_rows = None
test_data_columns = None
test_attrition_final = None

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = APP_ROOT + "/static"

app = Flask(__name__)
app.secret_key ="jfhjgbhk"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login",methods=['post'])
def login():
    UserName = request.form.get("UserName")
    password = request.form.get("password")
    if UserName == 'employer' and password == 'employer':
        session['role'] ='employer'
        return render_template("ehome.html")
    else:
        return render_template("message.html",msg='Invalid Login Details',color='text-danger')


@app.route("/ehome")
def ehome():
    return render_template("ehome.html")


@app.route("/logout")
def logout():
    session.clear()
    return render_template("index.html")


@app.route("/upload_train")
def upload_train():
    return render_template("upload_train.html")


@app.route("/train_upload1",methods=['post'])
def train_upload1():
    global train_data
    global data_rows
    global data_columns
    UploadTrain = request.files.get("UploadTrain")
    path = APP_ROOT + "/dataset/" + UploadTrain.filename
    UploadTrain.save(path)
    train_data = pd.read_csv(path)
    data_rows = train_data.values.tolist()
    data_columns = train_data.columns.tolist()
    return render_template("view_train_data.html",data_columns=data_columns, data_rows=data_rows, len=len)


@app.route("/train")
def train():
    global train_data
    train_data = pd.DataFrame(data_rows, columns=data_columns)
    train_data.info()
    print(train_data.head())
    print(train_data.info())

    train_data.drop(['Ename', 'EmployeeNumber'], axis=1, inplace=True)

    # statistical view of dataset
    train_data.describe()

    """EXPLORATORY DATA ANALYSIS"""

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric_df = train_data.select_dtypes(include=numerics)
    numeric_df.columns

    category_df = train_data.select_dtypes(include='object')
    category_df

    # # finding the correlation between the columns
    # corr = numeric_df.corr()
    #
    # sns.heatmap(corr)
    # plt.show()

    """There are high correlation between the columns

    ## Feature engineering and categorical encoding-Training
    """

    # Define a dictionary for the target mapping
    target_map = {'Yes': 1, 'No': 0}
    # Use the pandas apply method to numerically encode our attrition target variable
    train_data["Attrition_numerical"] = train_data["Attrition"].apply(lambda x: target_map[x])

    # Drop the Attrition_numerical column from attrition dataset first - Don't want to include that
    train_attrition = train_data.drop(['Attrition_numerical'], axis=1)

    ### Training data....
    # Empty list to store columns with categorical data
    categorical = []
    for col, value in train_attrition.items():
        if value.dtype == 'object':
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = train_attrition.columns.difference(categorical)

    # Store the categorical data in a dataframe called attrition_cat
    train_attrition_cat = train_attrition[categorical]
    train_attrition_cat = train_attrition_cat.drop(['Attrition'], axis=1)  # Dropping the target column

    train_attrition_cat = pd.get_dummies(train_attrition_cat)
    train_attrition_cat.head(3)

    # Store the numerical features to a dataframe attrition_num
    train_attrition_num = train_attrition[numerical]

    # Concat the two dataframes together columnwise
    train_attrition_final = pd.concat([train_attrition_num, train_attrition_cat], axis=1)

    # Define a dictionary for the target mapping
    target_map = {'Yes': 1, 'No': 0}
    # Use the pandas apply method to numerically encode our attrition target variable
    train_target = train_attrition["Attrition"].apply(lambda x: target_map[x])
    train_target.head(3)

    """## Naive Bayes classifier"""


    nb = GaussianNB()

    nb.fit(train_attrition_final.values, train_target)
    print('Model training completed............')
    return render_template("message.html",msg='Training Finished Successfully',color='text-success')

@app.route("/upload_test")
def upload_test():
    return render_template("upload_test.html")


@app.route("/upload_test1",methods=['post'])
def upload_test1():
    global test_data
    global test_data_rows
    global test_data_columns
    UploadTest = request.files.get("UploadTest")
    path = APP_ROOT + "/dataset/" + UploadTest.filename
    UploadTest.save(path)
    test_data = pd.read_csv(path)
    test_data_rows = test_data.values.tolist()
    test_data_columns = test_data.columns.tolist()
    return render_template("view_test_data.html", data_columns=test_data_columns, data_rows=test_data_rows, len=len)



@app.route("/test")
def test():
    global  test_data
    global test_attrition_final
    with open('model.pkl', 'rb') as f:
        nb = pickle.load(f)
    target_map = {'Yes': 1, 'No': 0}
    test_data = pd.read_csv('Emp_attrition_Test.csv')
    test_data.drop(['Ename', 'EmployeeNumber'], axis=1, inplace=True)

    test_data["Attrition_numerical"] = test_data["Attrition"].apply(lambda x: target_map[x])
    test_attrition = test_data.drop(['Attrition_numerical'], axis=1)

    # Testing data.....
    # Empty list to store columns with categorical data
    categorical = []
    for col, value in test_attrition.items():
        if value.dtype == 'object':
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = test_attrition.columns.difference(categorical)

    # Store the categorical data in a dataframe called attrition_cat
    test_attrition_cat = test_attrition[categorical]
    test_attrition_cat = test_attrition_cat.drop(['Attrition'], axis=1)  # Dropping the target column

    test_attrition_cat = pd.get_dummies(test_attrition_cat)
    test_attrition_cat.head(10)

    # Store the numerical features to a dataframe attrition_num
    test_attrition_num = test_attrition[numerical]

    # Concat the two dataframes together columnwise
    test_attrition_final = pd.concat([test_attrition_num, test_attrition_cat], axis=1)

    # Define a dictionary for the target mapping
    target_map = {'Yes': 1, 'No': 0}
    # Use the pandas apply method to numerically encode our attrition target variable
    test_target = test_attrition["Attrition"].apply(lambda x: target_map[x])
    test_target.head(3)

    results = nb.predict_proba(test_attrition_final)

    results_1 = results[:, 1]

    np.set_printoptions(suppress=True)
    len(results)
    # results

    print(np.argmax(results))

    test_attrition_final.shape

    predictions = nb.predict(test_attrition_final.values)
    print(predictions.shape)
    print(test_target.shape)

    from sklearn.metrics import accuracy_score
    model_accuracy = accuracy_score(test_target, nb.predict(test_attrition_final.values))
    print(model_accuracy)

    np.set_printoptions(precision=2, suppress=True)
    results_1 = results_1.tolist()

    test_data = pd.read_csv('Emp_attrition_Test.csv')
    test_data['results'] = results_1
    # test_data

    final_data = test_data.sort_values(by=['results'], ascending=False).head(5)[
        ['Ename', 'EmployeeNumber']].reset_index(drop=True)

    # print(final_data)
    data_rows = final_data.values.tolist()
    data_columns = final_data.columns.tolist()
    return render_template("testing_results.html",data_rows=data_rows,data_columns=data_columns,len=len)


@app.route("/update_data_row")
def update_data_row():
    global data_rows
    index = request.args.get("index")
    print(index)
    print(data_rows)
    if data_rows == None:
        return render_template("message.html",msg='Please Upload Dataset',color='text-primary')
    return render_template("update_data_row.html",data_row=data_rows[int(index)],index=index)



@app.route("/update_train_data",methods=['post'])
def update_train_data():
    global data_rows
    index = int(request.form.get("index"))
    # index = len(data_rows)
    Ename = request.form.get("Ename")
    Age = request.form.get("Age")
    NumCompaniesWorked = request.form.get("NumCompaniesWorked")
    Previous_company_experience = request.form.get("Previous company experience")
    Job_role_in_previous_company = request.form.get("Job role in previous company")
    Number_of_years_experience_in_prevoius_company = request.form.get("Number of years experience in prevoius company")
    Reason_for_leaving = request.form.get("Reason for leaving")
    Salary_in_previous_company = request.form.get("Salary in previous company")
    Present_expectation = request.form.get("Present expectation Y/N")
    Expectation_of_hike_in = request.form.get("Expectation of hike in %")
    Promotion = request.form.get("Promotion")
    New_designation_after_promotion = request.form.get("New designation after promotion")
    Attrition = request.form.get("Attrition")
    BusinessTravel = request.form.get("BusinessTravel")
    Department = request.form.get("Department")
    Education = request.form.get("Education")
    EmployeeNumber = request.form.get("EmployeeNumber")
    EnvironmentSatisfaction = request.form.get("EnvironmentSatisfaction")
    Gender = request.form.get("Gender")
    JobInvolvement = request.form.get("JobInvolvement")
    JobLevel = request.form.get("JobLevel")
    JobRole = request.form.get("JobRole")
    JobSatisfaction = request.form.get("JobSatisfaction")
    MaritalStatus = request.form.get("MaritalStatus")
    MonthlyIncome = request.form.get("MonthlyIncome")
    OverTime = request.form.get("OverTime")
    PerformanceRating = request.form.get("PerformanceRating")
    StockOptionLevel = request.form.get("StockOptionLevel")
    TotalWorkingYears = request.form.get("TotalWorkingYears")
    YearsAtCompany = request.form.get("YearsAtCompany")
    YearsWithCurrManager = request.form.get("YearsWithCurrManager")
    data_rows[index][0] = Ename
    data_rows[index][1] = Age
    data_rows[index][2] = NumCompaniesWorked
    data_rows[index][3] = Previous_company_experience
    data_rows[index][4] = Job_role_in_previous_company
    data_rows[index][5] = Number_of_years_experience_in_prevoius_company
    data_rows[index][6] = Reason_for_leaving
    data_rows[index][7] = Salary_in_previous_company
    data_rows[index][8] = Present_expectation
    data_rows[index][9] = Expectation_of_hike_in
    data_rows[index][10] = Promotion
    data_rows[index][11] = New_designation_after_promotion
    data_rows[index][12] = Attrition
    data_rows[index][13] = BusinessTravel
    data_rows[index][14] = Department
    data_rows[index][15] = Education
    data_rows[index][16] = EmployeeNumber
    data_rows[index][17] = EnvironmentSatisfaction
    data_rows[index][18] = Gender
    data_rows[index][19] = JobInvolvement
    data_rows[index][20] = JobLevel
    data_rows[index][21] = JobRole
    data_rows[index][22] = JobSatisfaction
    data_rows[index][23] = MaritalStatus
    data_rows[index][24] = MonthlyIncome
    data_rows[index][25] = OverTime
    data_rows[index][26] = PerformanceRating
    data_rows[index][27] = StockOptionLevel
    data_rows[index][28] = TotalWorkingYears
    data_rows[index][29] = YearsAtCompany
    data_rows[index][30] = YearsWithCurrManager
    return render_template("view_train_data.html",data_columns=data_columns, data_rows=data_rows, len=len)


@app.route("/add_train_data_row")
def add_train_data_row():
    return render_template("add_train_data_row.html")



@app.route("/add_train_data1",methods=['post'])
def add_train_data1():
    global data_rows
    # index = int(request.form.get("index"))
    index = len(data_rows)
    Ename = request.form.get("Ename")
    Age = request.form.get("Age")
    NumCompaniesWorked = request.form.get("NumCompaniesWorked")
    Previous_company_experience = request.form.get("Previous company experience")
    Job_role_in_previous_company = request.form.get("Job role in previous company")
    Number_of_years_experience_in_prevoius_company = request.form.get("Number of years experience in prevoius company")
    Reason_for_leaving = request.form.get("Reason for leaving")
    Salary_in_previous_company = request.form.get("Salary in previous company")
    Present_expectation = request.form.get("Present expectation Y/N")
    Expectation_of_hike_in = request.form.get("Expectation of hike in %")
    Promotion = request.form.get("Promotion")
    New_designation_after_promotion = request.form.get("New designation after promotion")
    Attrition = request.form.get("Attrition")
    BusinessTravel = request.form.get("BusinessTravel")
    Department = request.form.get("Department")
    Education = request.form.get("Education")
    EmployeeNumber = request.form.get("EmployeeNumber")
    EnvironmentSatisfaction = request.form.get("EnvironmentSatisfaction")
    Gender = request.form.get("Gender")
    JobInvolvement = request.form.get("JobInvolvement")
    JobLevel = request.form.get("JobLevel")
    JobRole = request.form.get("JobRole")
    JobSatisfaction = request.form.get("JobSatisfaction")
    MaritalStatus = request.form.get("MaritalStatus")
    MonthlyIncome = request.form.get("MonthlyIncome")
    OverTime = request.form.get("OverTime")
    PerformanceRating = request.form.get("PerformanceRating")
    StockOptionLevel = request.form.get("StockOptionLevel")
    TotalWorkingYears = request.form.get("TotalWorkingYears")
    YearsAtCompany = request.form.get("YearsAtCompany")
    YearsWithCurrManager = request.form.get("YearsWithCurrManager")
    data_row = [Ename,Age,NumCompaniesWorked,Previous_company_experience,Job_role_in_previous_company,Number_of_years_experience_in_prevoius_company,Reason_for_leaving,Salary_in_previous_company,Present_expectation,Expectation_of_hike_in,Promotion,New_designation_after_promotion,Attrition,BusinessTravel,Department,Education,EmployeeNumber,EnvironmentSatisfaction,
                Gender,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,OverTime,PerformanceRating,StockOptionLevel,TotalWorkingYears,YearsAtCompany,YearsWithCurrManager]
    data_rows.append(data_row)
    # data_rows[index][0] = Ename
    # data_rows[index][1] = Age
    # data_rows[index][2] = NumCompaniesWorked
    # data_rows[index][3] = Previous_company_experience
    # data_rows[index][4] = Job_role_in_previous_company
    # data_rows[index][5] = Number_of_years_experience_in_prevoius_company
    # data_rows[index][6] = Reason_for_leaving
    # data_rows[index][7] = Salary_in_previous_company
    # data_rows[index][8] = Present_expectation
    # data_rows[index][9] = Expectation_of_hike_in
    # data_rows[index][10] = Promotion
    # data_rows[index][11] = New_designation_after_promotion
    # data_rows[index][12] = Attrition
    # data_rows[index][13] = BusinessTravel
    # data_rows[index][14] = Department
    # data_rows[index][15] = Education
    # data_rows[index][16] = EmployeeNumber
    # data_rows[index][17] = EnvironmentSatisfaction
    # data_rows[index][18] = Gender
    # data_rows[index][19] = JobInvolvement
    # data_rows[index][20] = JobLevel
    # data_rows[index][21] = JobRole
    # data_rows[index][22] = JobSatisfaction
    # data_rows[index][23] = MaritalStatus
    # data_rows[index][24] = MonthlyIncome
    # data_rows[index][25] = OverTime
    # data_rows[index][26] = PerformanceRating
    # data_rows[index][27] = StockOptionLevel
    # data_rows[index][28] = TotalWorkingYears
    # data_rows[index][29] = YearsAtCompany
    # data_rows[index][30] = YearsWithCurrManager
    return render_template("view_train_data.html", data_columns=data_columns, data_rows=data_rows, len=len)


@app.route("/update_test_data_row")
def update_test_data_row():
    global test_data_rows
    index = request.args.get("index")
    print(index)
    if test_data_rows == None:
        return render_template("message.html",msg='Please Upload Dataset',color='text-primary')
    return render_template("update_test_data_row.html",data_row=test_data_rows[int(index)],index=index)

@app.route("/update_test_data",methods=['post'])
def update_test_data():
    global test_data_rows
    index = int(request.form.get("index"))
    # index = len(data_rows)
    Ename = request.form.get("Ename")
    Age = request.form.get("Age")
    NumCompaniesWorked = request.form.get("NumCompaniesWorked")
    Previous_company_experience = request.form.get("Previous company experience")
    Job_role_in_previous_company = request.form.get("Job role in previous company")
    Number_of_years_experience_in_prevoius_company = request.form.get("Number of years experience in prevoius company")
    Reason_for_leaving = request.form.get("Reason for leaving")
    Salary_in_previous_company = request.form.get("Salary in previous company")
    Present_expectation = request.form.get("Present expectation Y/N")
    Expectation_of_hike_in = request.form.get("Expectation of hike in %")
    Promotion = request.form.get("Promotion")
    New_designation_after_promotion = request.form.get("New designation after promotion")
    Attrition = request.form.get("Attrition")
    BusinessTravel = request.form.get("BusinessTravel")
    Department = request.form.get("Department")
    Education = request.form.get("Education")
    EmployeeNumber = request.form.get("EmployeeNumber")
    EnvironmentSatisfaction = request.form.get("EnvironmentSatisfaction")
    Gender = request.form.get("Gender")
    JobInvolvement = request.form.get("JobInvolvement")
    JobLevel = request.form.get("JobLevel")
    JobRole = request.form.get("JobRole")
    JobSatisfaction = request.form.get("JobSatisfaction")
    MaritalStatus = request.form.get("MaritalStatus")
    MonthlyIncome = request.form.get("MonthlyIncome")
    OverTime = request.form.get("OverTime")
    PerformanceRating = request.form.get("PerformanceRating")
    StockOptionLevel = request.form.get("StockOptionLevel")
    TotalWorkingYears = request.form.get("TotalWorkingYears")
    YearsAtCompany = request.form.get("YearsAtCompany")
    YearsWithCurrManager = request.form.get("YearsWithCurrManager")
    test_data_rows[index][0] = Ename
    test_data_rows[index][1] = Age
    test_data_rows[index][2] = NumCompaniesWorked
    test_data_rows[index][3] = Previous_company_experience
    test_data_rows[index][4] = Job_role_in_previous_company
    test_data_rows[index][5] = Number_of_years_experience_in_prevoius_company
    test_data_rows[index][6] = Reason_for_leaving
    test_data_rows[index][7] = Salary_in_previous_company
    test_data_rows[index][8] = Present_expectation
    test_data_rows[index][9] = Expectation_of_hike_in
    test_data_rows[index][10] = Promotion
    test_data_rows[index][11] = New_designation_after_promotion
    test_data_rows[index][12] = Attrition
    test_data_rows[index][13] = BusinessTravel
    test_data_rows[index][14] = Department
    test_data_rows[index][15] = Education
    test_data_rows[index][16] = EmployeeNumber
    test_data_rows[index][17] = EnvironmentSatisfaction
    test_data_rows[index][18] = Gender
    test_data_rows[index][19] = JobInvolvement
    test_data_rows[index][20] = JobLevel
    test_data_rows[index][21] = JobRole
    test_data_rows[index][22] = JobSatisfaction
    test_data_rows[index][23] = MaritalStatus
    test_data_rows[index][24] = MonthlyIncome
    test_data_rows[index][25] = OverTime
    test_data_rows[index][26] = PerformanceRating
    test_data_rows[index][27] = StockOptionLevel
    test_data_rows[index][28] = TotalWorkingYears
    test_data_rows[index][29] = YearsAtCompany
    test_data_rows[index][30] = YearsWithCurrManager
    return render_template("view_test_data.html", data_columns=test_data_columns, data_rows=test_data_rows, len=len)


@app.route("/add_test_data_row")
def add_test_data_row():
    return render_template("add_test_data_row.html")




@app.route("/add_test_data1",methods=['post'])
def add_test_data1():
    global test_data_rows
    # index = int(request.form.get("index"))
    index = len(test_data_rows)
    Ename = request.form.get("Ename")
    Age = request.form.get("Age")
    NumCompaniesWorked = request.form.get("NumCompaniesWorked")
    Previous_company_experience = request.form.get("Previous company experience")
    Job_role_in_previous_company = request.form.get("Job role in previous company")
    Number_of_years_experience_in_prevoius_company = request.form.get("Number of years experience in prevoius company")
    Reason_for_leaving = request.form.get("Reason for leaving")
    Salary_in_previous_company = request.form.get("Salary in previous company")
    Present_expectation = request.form.get("Present expectation Y/N")
    Expectation_of_hike_in = request.form.get("Expectation of hike in %")
    Promotion = request.form.get("Promotion")
    New_designation_after_promotion = request.form.get("New designation after promotion")
    Attrition = request.form.get("Attrition")
    BusinessTravel = request.form.get("BusinessTravel")
    Department = request.form.get("Department")
    Education = request.form.get("Education")
    EmployeeNumber = request.form.get("EmployeeNumber")
    EnvironmentSatisfaction = request.form.get("EnvironmentSatisfaction")
    Gender = request.form.get("Gender")
    JobInvolvement = request.form.get("JobInvolvement")
    JobLevel = request.form.get("JobLevel")
    JobRole = request.form.get("JobRole")
    JobSatisfaction = request.form.get("JobSatisfaction")
    MaritalStatus = request.form.get("MaritalStatus")
    MonthlyIncome = request.form.get("MonthlyIncome")
    OverTime = request.form.get("OverTime")
    PerformanceRating = request.form.get("PerformanceRating")
    StockOptionLevel = request.form.get("StockOptionLevel")
    TotalWorkingYears = request.form.get("TotalWorkingYears")
    YearsAtCompany = request.form.get("YearsAtCompany")
    YearsWithCurrManager = request.form.get("YearsWithCurrManager")
    test_data_row = [Ename,Age,NumCompaniesWorked,Previous_company_experience,Job_role_in_previous_company,Number_of_years_experience_in_prevoius_company,Reason_for_leaving,Salary_in_previous_company,Present_expectation,Expectation_of_hike_in,Promotion,New_designation_after_promotion,Attrition,BusinessTravel,Department,Education,EmployeeNumber,EnvironmentSatisfaction,
                Gender,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,OverTime,PerformanceRating,StockOptionLevel,TotalWorkingYears,YearsAtCompany,YearsWithCurrManager]
    test_data_rows.append(test_data_row)
    # data_rows[index][0] = Ename
    # data_rows[index][1] = Age
    # data_rows[index][2] = NumCompaniesWorked
    # data_rows[index][3] = Previous_company_experience
    # data_rows[index][4] = Job_role_in_previous_company
    # data_rows[index][5] = Number_of_years_experience_in_prevoius_company
    # data_rows[index][6] = Reason_for_leaving
    # data_rows[index][7] = Salary_in_previous_company
    # data_rows[index][8] = Present_expectation
    # data_rows[index][9] = Expectation_of_hike_in
    # data_rows[index][10] = Promotion
    # data_rows[index][11] = New_designation_after_promotion
    # data_rows[index][12] = Attrition
    # data_rows[index][13] = BusinessTravel
    # data_rows[index][14] = Department
    # data_rows[index][15] = Education
    # data_rows[index][16] = EmployeeNumber
    # data_rows[index][17] = EnvironmentSatisfaction
    # data_rows[index][18] = Gender
    # data_rows[index][19] = JobInvolvement
    # data_rows[index][20] = JobLevel
    # data_rows[index][21] = JobRole
    # data_rows[index][22] = JobSatisfaction
    # data_rows[index][23] = MaritalStatus
    # data_rows[index][24] = MonthlyIncome
    # data_rows[index][25] = OverTime
    # data_rows[index][26] = PerformanceRating
    # data_rows[index][27] = StockOptionLevel
    # data_rows[index][28] = TotalWorkingYears
    # data_rows[index][29] = YearsAtCompany
    # data_rows[index][30] = YearsWithCurrManager
    return render_template("view_test_data.html", data_columns=test_data_columns, data_rows=test_data_rows, len=len)


@app.route("/Predictions")
def Predictions():
    # displaying all the results
    global test_data
    global test_attrition_final
    if test_data is None:
        return render_template("message.html",msg='Upload Dataset',color='text-primary')
    predictions = nb.predict(test_attrition_final.values)

    test_data['new'] = np.where(test_data['results'] <= 0.5, 0, 1)
    test_data
    test_data['predictions'] = predictions
    test_data
    # Define a dictionary for the target mapping
    target_map = {1: 'Yes', 0: 'No'}
    # Use the pandas apply method to numerically encode our attrition target variable
    test_data["attrition_status"] = test_data["predictions"].apply(lambda x: target_map[x])
    print(test_data[['Ename', 'EmployeeNumber', 'attrition_status']])
    # print(test_data)

    test_data__row_results = test_data.values.tolist()
    test_data_column_result = test_data.columns.tolist()
    return render_template("Predictions.html",data_rows=test_data__row_results,data_columns=test_data_column_result,len=len)

@app.route("/get_graphs")
def get_graphs():
    # data = pd.read_csv('emp_attrition_raw.csv')
    #
    # # dataset overview
    # data.info()
    #
    # # sample data
    # data.head(5)
    #
    # # Define a dictionary for the target mapping
    # target_map = {'Yes': 1, 'No': 0}
    # # Use the pandas apply method to numerically encode our attrition target variable
    # data["Attrition_numerical"] = data["Attrition"].apply(lambda x: target_map[x])
    #
    # # #monthly income vs attrition
    # # sns.boxplot(data=data, x='Attrition_numerical',y='MonthlyIncome')
    # # plt.show()
    #
    # # #attrition genderwise
    # # sns.countplot(x='Gender')
    # #
    # # plt.style.use('seaborn-pastel')
    # # plt.rcParams['figure.figsize'] = (4,4)
    #
    # sns.countplot(x=data['Attrition'], hue='BusinessTravel', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by BusinessTravel")
    # plt.savefig("static/fig1.png")
    # # plt.show()
    #
    # plt.style.use('seaborn-pastel')
    # plt.rcParams['figure.figsize'] = (8, 6)
    # sns.countplot(x=data['Attrition'], hue='Reason_for_leaving', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by Reason for leaving ")
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    # plt.savefig("static/fig2.png")
    # # plt.show()
    #
    # plt.style.use('seaborn-pastel')
    # plt.rcParams['figure.figsize'] = (6, 4)
    # sns.countplot(x=data['Attrition'], hue='PreviousCompanyExperience', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by PreviousCompanyExperience ")
    # plt.savefig("static/fig3.png")
    # # plt.show()
    #
    # plt.style.use('seaborn-pastel')
    # plt.rcParams['figure.figsize'] = (6, 4)
    # sns.countplot(x=data['Attrition'], hue='Gender', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by Gender")
    # plt.savefig("static/fig4.png")
    # # plt.show()
    #
    # plt.style.use('seaborn-pastel')
    # plt.rcParams['figure.figsize'] = (6, 4)
    # sns.countplot(x=data['Attrition'], hue='EnvironmentSatisfaction', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by EnvironmentSatisfaction")
    # plt.savefig("static/fig5.png")
    # # plt.show()
    #
    # plt.style.use('seaborn-pastel')
    # plt.rcParams['figure.figsize'] = (6, 4)
    # sns.countplot(x=data['Attrition'], hue='Expectation_of_hike', data=data, palette='PuRd').set_title(
    #     "Number of Attritions Reported by Expectation_of_hike")
    # plt.savefig("static/fig6.png")
    # # plt.show()
    return render_template("get_graphs.html")

app.run(debug=True)