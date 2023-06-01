{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive"
      ],
      "metadata": {
        "id": "PhcljhXj9r81"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "id": "DdHt7bkE9r6B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.datasets import make_classification\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import accuracy_score"
      ],
      "metadata": {
        "id": "05iC232S9r3E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O0xZrMVc9fD9"
      },
      "outputs": [],
      "source": [
        "file_pathtr ='/content/drive/MyDrive/NSL-KDD dataset/KDDTrain+.txt'\n",
        "train_data = pd.read_csv(file_pathtr, header=None)\n",
        "file_pathte='/content/drive/MyDrive/NSL-KDD dataset/KDDTest+.txt'\n",
        "test_data = pd.read_csv(file_pathte, header=None)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',\n",
        "                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', \n",
        "                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',\n",
        "                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', \n",
        "                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count','dst_host_srv_count',\n",
        "                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', \n",
        "                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'duration']"
      ],
      "metadata": {
        "id": "wP_ov0b698VI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.columns = column_names\n",
        "test_data.columns = column_names"
      ],
      "metadata": {
        "id": "Guea2uOa-Cw0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.head()"
      ],
      "metadata": {
        "id": "D3JpcfDx-OFM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_data.head()"
      ],
      "metadata": {
        "id": "0N1Pua04-OBc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = train_data.drop('label', axis=1)\n",
        "y_train = train_data['label']\n",
        "X_test = test_data.drop('label', axis=1)\n",
        "y_test = test_data['label']"
      ],
      "metadata": {
        "id": "uSsUVSSk-N_t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = pd.get_dummies(X_train, columns=['protocol_type', 'service', 'flag'])\n",
        "X_test = pd.get_dummies(X_test, columns=['protocol_type', 'service', 'flag'])"
      ],
      "metadata": {
        "id": "AHtzo_N6-N8p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import MinMaxScaler\n",
        "scaler = MinMaxScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.fit_transform(X_test)"
      ],
      "metadata": {
        "id": "nZMuHzb5-N68"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "def le(df):\n",
        "    df = pd.DataFrame(df)  # Convert NumPy array to pandas DataFrame\n",
        "    for col in df.columns:\n",
        "        if df[col].dtype == 'object':\n",
        "            label_encoder = LabelEncoder()\n",
        "            df[col] = label_encoder.fit_transform(df[col])\n",
        "    return df.values  # Convert back to NumPy array if necessary\n",
        "\n",
        "le(X_train)\n",
        "le(X_test)"
      ],
      "metadata": {
        "id": "pUsSdAwE-N3t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train_df = pd.DataFrame(X_train)\n",
        "X_test_df = pd.DataFrame(X_test)\n",
        "test_cols = set(X_train_df.columns)\n",
        "train_cols = set(X_test_df.columns)\n",
        "missing_cols = train_cols.difference(test_cols)\n",
        "extra_cols = test_cols.difference(train_cols)\n",
        "print(\"Missing columns:\", missing_cols)\n",
        "print(\"Extra columns:\", extra_cols)"
      ],
      "metadata": {
        "id": "Lzh-gtdJ-N2C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for col in missing_cols:\n",
        "    X_test_df[col] = 0\n",
        "X_test_df.fillna(0, inplace=True)"
      ],
      "metadata": {
        "id": "MQfsx5Cg_AlI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for col in extra_cols:\n",
        "  if col in X_train_df.columns:\n",
        "    X_train_df.drop(col, axis=1, inplace=True)"
      ],
      "metadata": {
        "id": "d9U6dOxX_Ajj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "import itertools\n",
        "\n",
        "tree_clf = DecisionTreeClassifier(random_state=42)\n",
        "\n",
        "tree_clf.fit(X_train_df, y_train)\n",
        "\n",
        "rfe = RFE(tree_clf, n_features_to_select=10)\n",
        "rfe = rfe.fit(X_train_df, y_train)\n",
        "\n",
        "feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.support_, X_train.columns)]\n",
        "selected_features = [v for i, v in feature_map if i]\n"
      ],
      "metadata": {
        "id": "uPPSMGINAqzX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = X_train[selected_features]"
      ],
      "metadata": {
        "id": "9HeY377w_AgD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = svm_clf.predict(X_test_df)"
      ],
      "metadata": {
        "id": "9UCgjMK-_Aei"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred))\n",
        "print(\"Accuracy Score:\")\n",
        "print(accuracy_score(y_test, y_pred))"
      ],
      "metadata": {
        "id": "CxDXNFW9_i08"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import f1_score, precision_score, recall_score\n",
        "f= f1_score(y_test, y_pred,average=\"micro\")\n",
        "p= precision_score(y_test, y_pred,average=\"weighted\",zero_division=1.0)\n",
        "r= recall_score(y_test, y_pred,average=\"weighted\",zero_division=1.0)\n",
        "print(\"f1_score = \",f)\n",
        "print(\"Precision Score = \",p)\n",
        "print(\"recall score = \",r)"
      ],
      "metadata": {
        "id": "YGcxO8Ws_wCk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import classification_report\n",
        "print(\"SVM Classifier:\")\n",
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "id": "pe3Q_ZS4_vw2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "import pickle \n",
        "joblib.dump(svm_clf, \"svm.pkl\")"
      ],
      "metadata": {
        "id": "dAP_ZLefAMI_"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}