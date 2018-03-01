
def TotalFeatureGenerator(data,target_var):
    # Creating iterator

    keys = data.keys()
    print("The features are the following:\n", keys)
    feature_list = data.columns.values.tolist()
    feature_list.remove(str(target_var))
    np_feature_list = np.array(feature_list)
    print(np_feature_list.size)

    target = data[str(target_var)].values
    input_data = data.drop(str(target_var), axis=1).values

    data_train, data_test, target_train, target_test = train_test_split(input_data, target, test_size= 0.3, random_state= 25)
    rf = RandomForestClassifier()
    rf.fit(data_train,target_train)

    importance = (rf.feature_importances_)*100
    np_importance = np.array(importance)
    print(np_importance.size)

    np_2Darray = np.column_stack((np_feature_list,np_importance))
    df_feature_list = pd.DataFrame(data=np_2Darray)
    df_feature_list.columns = ["0", "1"]
    print(df_feature_list.dtypes)

    relevant = []

    try:
        a = df_feature_list.where(df_feature_list["1"]>2)
        print(a.head())
    except:
        print("cant hold condition")
