# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function


# 1. Exploratory Data Analysis
import numpy as np
import LinearRegression as LinearRegression
import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

df = pd.read_csv("witches.csv")

check_df(df)


miss_cols = ["Hair Color", "Eyebrows", "Necklace", "Hat", "Hair (Back)", "Face Markings", "Facewear", "Hair Topper", "Back Item", "Earrings", "Forehead Jewelry", "Hair (Middle)", "Mask", "Outerwear"]
for col in miss_cols:
    df.drop(col, axis=1, inplace=True)


# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

# for col in num_cols:
#     num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)


# Target ile sayısal değişkenlerin incelemesi
# for col in num_cols:
#     target_summary_with_num(df, "sales_price", col)



################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.head()

# Değişken isimleri büyütmek
df.columns = [col.upper() for col in df.columns]

# Feature Engineering:
df["sales_price"] = df["last_sale.total_price"]/1000000000000000000*df["last_sale.payment_token.usd_price"]
df = df.loc[~df.sales_price.isnull()]


check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

cardinal = ["id", "token_id"]
num_cols = [col for col in num_cols if col not in cardinal]


df.drop(["id", 'name', 'description', "owner.user.username", 'external_link', 'permalink', 'token_metadata', 'token_id', 'owner.address','last_sale.transaction.timestamp'], axis=1, inplace=True)

df.drop(["Eye Style", "Eye Color", "Mouth", "Top"], axis=1, inplace=True)


for col in cat_cols:
    cat_summary(df, col)


# for col in cat_cols:
#     target_summary_with_cat(df, "sales_price", col)


categorical_cols = ["Skin Tone","Rising Sign", "Body Shape", "Moon Sign", "Sun Sign", "Archetype of Power", 'Hair (Front)', "Background"]
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df,categorical_cols,True)


check_df(df)

df.columns = [col.upper() for col in df.columns]

# Son güncel değişken türlerimi tutuyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
# cat_cols = [col for col in cat_cols if "OUTCOME" not in col]


for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

for col in num_cols:
    replace_with_thresholds(df,col)

df.drop(["id",'name', 'description', 'external_link', 'permalink', 'token_metadata', 'token_id', 'owner.address','last_sale.transaction.timestamp', "last_sale.total_price"], axis=1, inplace=True)


# Encoding:
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

df.drop(["Eye Style", "Eye Color", "Mouth", "Top"],axis=1, inplace=True)


# one-hot encoding:
categorical_cols = ["Skin Tone","Rising Sign", "Body Shape", "Moon Sign", "Sun Sign", "Archetype of Power", "Hair (Front)", "Background"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df,categorical_cols,True)


# Standartlaştırma
num_cols = [col for col in num_cols if col not in ["sales_price"]]
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


X = df.drop("sales_price", axis=1)
y = df["sales_price"]

check_df(X)

df.head(1)

######################################################
# 3. Base Models
######################################################
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# linearregresyon:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
lr = LinearRegression().fit(x_train, y_train)

y_pred = lr.predict(x_test)

lr_score = r2_score(y_test, y_pred)
lr_rmse = mean_squared_error(y_test, y_pred, squared = False)
print("R2 Score : ", lr_score)
print("RMSE : ", lr_rmse)



# randomforsts:
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor().fit(x_train, y_train)

y_pred = rfr.predict(x_test)
rfr_score = r2_score(y_test, y_pred)
rfr_rmse = mean_squared_error(y_test, y_pred, squared = False)
print("R2 Score : ", rfr_score)
print("RMSE : ", rfr_rmse)



# target categorical ise:
# def base_models(X, y, scoring="roc_auc"):
#     print("Base Models....")
#     classifiers = [('LR', LogisticRegression()),
#                    ('KNN', KNeighborsClassifier()),
#                    ("SVC", SVC()),
#                    ("CART", DecisionTreeClassifier()),
#                    ("RF", RandomForestClassifier()),
#                    ('Adaboost', AdaBoostClassifier()),
#                    ('GBM', GradientBoostingClassifier()),
#                    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
#                    ('LightGBM', LGBMClassifier()),
#                    # ('CatBoost', CatBoostClassifier(verbose=False))
#                    ]
#
#     for name, classifier in classifiers:
#         cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
#         print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
#
# base_models(X, y, scoring="accuracy")



######################################################
# 4. Automated Hyperparameter Optimization
######################################################
gbm_model = GradientBoostingRegressor(random_state=17)
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}



gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))



# Target categorical:
# knn_params = {"n_neighbors": range(2, 50)}
#
# cart_params = {'max_depth': range(1, 20),
#                "min_samples_split": range(2, 30)}
#
# rf_params = {"max_depth": [8, 15, None],
#              "max_features": [5, 7, "auto"],
#              "min_samples_split": [15, 20],
#              "n_estimators": [200, 300]}
#
# xgboost_params = {"learning_rate": [0.1, 0.01],
#                   "max_depth": [5, 8],
#                   "n_estimators": [100, 200]}
#
# lightgbm_params = {"learning_rate": [0.01, 0.1],
#                    "n_estimators": [300, 500]}
#
#
# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
#                ('LightGBM', LGBMClassifier(), lightgbm_params)]

# def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
#     print("Hyperparameter Optimization....")
#     best_models = {}
#     for name, classifier, params in classifiers:
#         print(f"########## {name} ##########")
#         cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
#
#         gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
#         final_model = classifier.set_params(**gs_best.best_params_)
#
#         cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
#         print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
#         best_models[name] = final_model
#     return best_models

# best_models = hyperparameter_optimization(X, y)



######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)


######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)