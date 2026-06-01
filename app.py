{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0599a48-1b24-4f6c-88ba-a8485cc29892",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ed19c55-0e6f-487e-9c7f-dcdd26e50f06",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"loan_approval_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c708c5c-8a24-4bf7-b9c3-35673267b194",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()\n",
    "df.info()\n",
    "df.isnull().sum()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f60919b-8274-4c90-89aa-1c2ead508149",
   "metadata": {},
   "source": [
    "# Handle Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf70ad4f-49f6-4769-93a1-bdfeb1ffb313",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_cols = df.select_dtypes(include=[\"object\"]).columns\n",
    "numerical_cols = df.select_dtypes(include=[\"number\"]).columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "814bfe60-647a-4b8c-9d76-211d5bff1dcd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "num_imp = SimpleImputer(strategy=\"mean\")\n",
    "df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4565875d-b068-4f60-ae31-0041a49851f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_imp = SimpleImputer(strategy=\"most_frequent\")\n",
    "df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e7cc717-5942-4dac-a95a-f22c7c629e82",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be4c688b-94fb-413c-b69a-35a256460470",
   "metadata": {},
   "source": [
    "# EDA - exploratory data analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "750ca54e-b84e-4ea1-9ce9-6a33d93dba52",
   "metadata": {},
   "outputs": [],
   "source": [
    "# how balanced our classes are?\n",
    "\n",
    "classes_count = df[\"Loan_Approved\"].value_counts()\n",
    "\n",
    "plt.pie(classes_count, labels=[\"No\", \"Yes\"], autopct=\"%1.1f%%\")\n",
    "plt.title(\"Is Loan approved or not?\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56b5797c-6211-4f7e-b303-05371e18750f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# analyze categories\n",
    "\n",
    "# gender_cnt = df[\"Gender\"].value_counts()\n",
    "# ax = sns.barplot(gender_cnt)\n",
    "# ax.bar_label(ax.containers[0])\n",
    "\n",
    "edu_cnt = df[\"Education_Level\"].value_counts()\n",
    "ax = sns.barplot(edu_cnt)\n",
    "ax.bar_label(ax.containers[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5b27eda-b676-4f9e-8aaf-4e70cfbb9d53",
   "metadata": {},
   "outputs": [],
   "source": [
    "# analyze income\n",
    "\n",
    "sns.histplot(\n",
    "    data = df,\n",
    "    x = \"Applicant_Income\",\n",
    "    bins=20\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52b8d0cb-509a-43e6-9902-638b77118337",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.histplot(\n",
    "    data = df,\n",
    "    x = \"Coapplicant_Income\",\n",
    "    bins=20\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28cb6bc5-b073-441a-a834-23bafcf6749a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# outliers - box plots\n",
    "\n",
    "sns.boxplot(\n",
    "    data=df,\n",
    "    x=\"Loan_Approved\",\n",
    "    y=\"Applicant_Income\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38c092cf-5a70-4a79-b200-c375013d173e",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2)\n",
    "\n",
    "sns.boxplot(ax=axes[0, 0], data=df, x=\"Loan_Approved\",y=\"Applicant_Income\")\n",
    "sns.boxplot(ax=axes[0, 1], data=df, x=\"Loan_Approved\",y=\"Credit_Score\")\n",
    "sns.boxplot(ax=axes[1, 0], data=df, x=\"Loan_Approved\",y=\"DTI_Ratio\")\n",
    "sns.boxplot(ax=axes[1, 1], data=df, x=\"Loan_Approved\",y=\"Savings\")\n",
    "\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d975229-dc92-4823-bbf7-9e21161cdda0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Credit Score with Loan Approved\n",
    "\n",
    "sns.histplot(\n",
    "    data=df,\n",
    "    x=\"Credit_Score\",\n",
    "    hue=\"Loan_Approved\",\n",
    "    bins=20,\n",
    "    multiple=\"dodge\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0365d234-228b-4e70-b9b5-dcb6680d4215",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.histplot(\n",
    "    data=df,\n",
    "    x=\"Applicant_Income\",\n",
    "    hue=\"Loan_Approved\",\n",
    "    bins=20,\n",
    "    multiple=\"dodge\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99e8253e-a344-4d8b-a911-0c540932607d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove Applicant Id\n",
    "df = df.drop(\"Applicant_ID\", axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f4a9076-fd33-45bf-a51f-91e7a21a6de9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b32c5d3a-3031-48e6-95fd-31e8344b5f65",
   "metadata": {},
   "source": [
    "# Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "488efb11-2046-4a2f-91f2-7aaf915bec6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()\n",
    "df.columns\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8a033f2-9a49-4492-a1d7-797263cba312",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
    "\n",
    "le = LabelEncoder()\n",
    "df[\"Education_Level\"] = le.fit_transform(df[\"Education_Level\"])\n",
    "df[\"Loan_Approved\"] = le.fit_transform(df[\"Loan_Approved\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ebe8192-4023-4db7-bbde-365180d7e924",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f481e15d-48f1-4c27-a9d7-10cb84e57a4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = [\"Employment_Status\", \"Marital_Status\", \"Loan_Purpose\", \"Property_Area\", \"Gender\", \"Employer_Category\"]\n",
    "\n",
    "ohe = OneHotEncoder(drop=\"first\", sparse_output=False, handle_unknown=\"ignore\")\n",
    "\n",
    "encoded = ohe.fit_transform(df[cols])\n",
    "\n",
    "encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)\n",
    "\n",
    "df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "214f39a4-d619-4ccb-b5e9-15e389e6593b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5dde863c-d4d2-41f1-a592-31a30564776d",
   "metadata": {},
   "source": [
    "# Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68c0bb60-046d-4f91-8277-0b26af6416e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = df.select_dtypes(include=\"number\")\n",
    "corr_matrix = num_cols.corr()\n",
    "\n",
    "plt.figure(figsize=(15, 8))\n",
    "sns.heatmap(\n",
    "    corr_matrix,\n",
    "    annot=True,\n",
    "    fmt=\".2f\",\n",
    "    cmap=\"coolwarm\"\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e56b78e-6415-4201-90c6-c196e47de082",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols.corr()[\"Loan_Approved\"].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96738500-b998-469a-a018-d4beb3e711ad",
   "metadata": {},
   "source": [
    "# Train-Test-Split + Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0c65852-9bf3-4733-bc80-791641d454ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"Loan_Approved\", axis=1)\n",
    "y = df[\"Loan_Approved\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "300e21e8-80cb-40f9-b06f-85954c985d89",
   "metadata": {},
   "outputs": [],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15cd99c9-91cf-4c90-8ff4-3425f9ce0a99",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43629831-1222-4bac-aa9b-896bc8db2c03",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0ac81d1-b93a-40a8-8cd3-728419ac37f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "scaler = StandardScaler()\n",
    "\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75bb7751-d69b-46fe-9ec9-569881e79d0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_scaled"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "789b9506-fe44-4fa9-bb67-ee0445936179",
   "metadata": {},
   "source": [
    "# Train & Evaluate Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d53ee94-1e1f-4a17-9e84-bccc8f1d3cf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic regression\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "log_model = LogisticRegression()\n",
    "log_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = log_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Logistic Regression Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d384892-2750-4238-9cbe-7aefa063bb9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# KNN\n",
    "\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "knn_model = KNeighborsClassifier(n_neighbors=5)\n",
    "knn_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = knn_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"KNN Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a54f45b9-5ea0-46f8-bc24-c9bbf1a9174a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Naive Bayes\n",
    "\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "nb_model = GaussianNB()\n",
    "nb_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = nb_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Naive Bayes Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd80ed9e-6ea5-4c7a-8548-99f2432c7759",
   "metadata": {},
   "source": [
    "## Best Model on the basis of Precision => Naive Bayes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eae30b0c-bbbd-4e8a-9a8b-37dd35a038bc",
   "metadata": {},
   "source": [
    "# Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b039970-fbba-4145-a2f2-896e53723a5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add or Tranform features\n",
    "df[\"DTI_Ratio_sq\"] = df[\"DTI_Ratio\"] ** 2\n",
    "df[\"Credit_Score_sq\"] = df[\"Credit_Score\"] ** 2\n",
    "\n",
    "# df[\"Applicant_Income_log\"] = np.log1p(df[\"Applicant_Income\"])\n",
    "\n",
    "X = df.drop(columns=[\"Loan_Approved\", \"Credit_Score\", \"DTI_Ratio\"])\n",
    "y = df[\"Loan_Approved\"]\n",
    "\n",
    "# Train test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Scaling\n",
    "scaler = StandardScaler()\n",
    "\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13d2ef30-249a-4abd-be4d-8a14873d4e2c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d78150b-9b40-46ff-a586-fd5f7221ce18",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic regression\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "log_model = LogisticRegression()\n",
    "log_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = log_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Logistic Regression Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d16df3c-9545-4ad0-8fcc-1f007bd94249",
   "metadata": {},
   "outputs": [],
   "source": [
    "# KNN\n",
    "\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "knn_model = KNeighborsClassifier(n_neighbors=5)\n",
    "knn_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = knn_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"KNN Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e2c801f-a3ca-45b2-bae6-35cd7aeb2ddb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Naive Bayes\n",
    "\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "nb_model = GaussianNB()\n",
    "nb_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = nb_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Naive Bayes Model\")\n",
    "print(\"Precision: \", precision_score(y_test, y_pred))\n",
    "print(\"Recall: \", recall_score(y_test, y_pred))\n",
    "print(\"F1 score: \", f1_score(y_test, y_pred))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, y_pred))\n",
    "print(\"CM: \", confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "843dab62-a1ff-4d0e-854e-a6ac83edba92",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
