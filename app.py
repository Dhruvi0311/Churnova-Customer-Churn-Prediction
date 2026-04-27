import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from flask import Flask, request, jsonify, render_template
import base64
import io
import matplotlib
matplotlib.use('Agg')

# Try to import data science libraries with error handling
try:
    import numpy as np
    print(" NumPy imported successfully")
except Exception as e:
    print(f" NumPy import failed: {e}")
    np = None

try:
    import pandas as pd
    print(" Pandas imported successfully")
except Exception as e:
    print(f" Pandas import failed: {e}")
    pd = None

try:
    import pickle
    import joblib
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, Rectangle
    print(" Matplotlib and joblib imported successfully")
except Exception as e:
    print(f" Matplotlib/joblib import failed: {e}")
    plt = None



import time

app = Flask(__name__)

# Load models with error handling
model = None
survmodel = None

if pickle is not None:
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        print(" Main model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        model = None

    try:
        survmodel = pickle.load(open('survivemodel.pkl', 'rb'))
        print("Survival model loaded successfully")
    except Exception as e:
        print(f" Error loading survival model: {e}")
        survmodel = None
else:
    print(" Pickle not available, cannot load models")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Check if required libraries are available
    if np is None:
        return render_template('index.html', prediction_text='NumPy not available. Cannot process prediction.', 
                          url_1='', url_2='', url_3='', url_4='')

    gender = 0
    if request.form["gender"] == "1":
        gender = 1
    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1
    Partner = 0
    if 'Partner' in request.form:
        Partner = 1
    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1
    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = int(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges*Tenure

    PhoneService = 0
    if 'PhoneService' in request.form:
        PhoneService = 1

    MultipleLines = 0
    if 'MultipleLines' in request.form and PhoneService == 1:
        MultipleLines = 1

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if request.form["InternetService"] == "0":
        InternetService_No = 1
    elif request.form["InternetService"] == "2":
        InternetService_Fiberoptic = 1

    OnlineSecurity = 0
    if 'OnlineSecurity' in request.form and InternetService_No == 0:
        OnlineSecurity = 1

    OnlineBackup = 0
    if 'OnlineBackup' in request.form and InternetService_No == 0:
        OnlineBackup = 1

    DeviceProtection = 0
    if 'DeviceProtection' in request.form and InternetService_No == 0:
        DeviceProtection = 1

    TechSupport = 0
    if 'TechSupport' in request.form and InternetService_No == 0:
        TechSupport = 1

    StreamingTV = 0
    if 'StreamingTV' in request.form and InternetService_No == 0:
        StreamingTV = 1

    StreamingMovies = 0
    if 'StreamingMovies' in request.form and InternetService_No == 0:
        StreamingMovies = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if request.form["Contract"] == "1":
        Contract_Oneyear = 1
    elif request.form["Contract"] == "2":
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if request.form["PaymentMethod"] == "1":
        PaymentMethod_CreditCard = 1
    elif request.form["PaymentMethod"] == "2":
        PaymentMethod_ElectronicCheck = 1
    elif request.form["PaymentMethod"] == "3":
        PaymentMethod_MailedCheck = 1

    features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
       InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
       PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
       'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = np.array(features).reshape(1, -1)
    
    # Check if model is loaded before using it
    if model is None:
        return render_template('index.html', prediction_text='Model not available. Please check model file.', 
                          url_1='', url_2='', url_3='', url_4='')
    
    prediction = model.predict_proba(final_features)
    output = prediction[0,1]

    # SAFE VARIABLES
    max_life = Tenure * 2
    CLTV = max_life * MonthlyCharges

    churn_prob = round(output, 2)
    clv = int(CLTV)
    
    CLTV = max_life * MonthlyCharges

    # Gauge plot
    def degree_range(n):
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
              colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):

        N = len(labels)
        colors = colors[::-1]


        """
        begins the plotting
        """

        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)

        labels = labels[::-1]

        """
        plots the sectors and the arcs
        """
        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]


        """
        set the labels (e.g. 'LOW','MEDIUM',...)
        """

        for mid, lab in zip(mid_points, labels):

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation = rot_text(mid))

        """
        set the bottom banner and the title
        """
        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability,2).astype(str), horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        """
        plots the arrow now
        """

        pos = (1-Probability)*180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                     width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        """
        removes frame and ticks, and makes axis equal and tight
        """

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format = 'png')
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability=output)

    # ===================== GRAPHS =====================

    imp_url = ''
    contrib_url = ''
    prob_url = ''
    value_url = ''

    try:
        importances = model.feature_importances_

        # Feature Importance
        imp_img = io.BytesIO()
        indices = np.argsort(importances)[-8:]
        plt.figure(figsize=(6,4))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), np.array(columns)[indices])
        plt.title("Top Features Affecting Churn")
        plt.savefig(imp_img, format='png')
        imp_img.seek(0)
        imp_url = base64.b64encode(imp_img.getvalue()).decode()

        # Contribution
        contrib_img = io.BytesIO()
        vals = final_features.flatten()
        top_idx = np.argsort(np.abs(importances))[-6:]
        plt.figure(figsize=(6,4))
        plt.barh(range(len(top_idx)), vals[top_idx])
        plt.yticks(range(len(top_idx)), np.array(columns)[top_idx])
        plt.title("Customer Contribution")
        plt.savefig(contrib_img, format='png')
        contrib_img.seek(0)
        contrib_url = base64.b64encode(contrib_img.getvalue()).decode()

        # Probability
        prob_img = io.BytesIO()
        plt.figure()
        plt.bar(['Stay','Churn'], [1-output, output])
        plt.title("Churn Probability")
        plt.savefig(prob_img, format='png')
        prob_img.seek(0)
        prob_url = base64.b64encode(prob_img.getvalue()).decode()

        # Value vs Risk
        value_img = io.BytesIO()
        plt.figure()
        plt.scatter([output], [CLTV])
        plt.title("Risk vs Value")
        plt.savefig(value_img, format='png')
        value_img.seek(0)
        value_url = base64.b64encode(value_img.getvalue()).decode()

    except Exception as e:
        print("Graph error:", e)

    if output < 0.3:
        conclusion = (
        "The customer shows a low likelihood of churn, indicating strong engagement and satisfaction with the current services. "
        "This suggests that the existing offerings and support are effective. The company should focus on maintaining service quality "
        "and strengthening customer relationships through loyalty programs or personalized engagement."
    )

    elif output < 0.6:
        conclusion = (
        "The customer falls under a moderate churn risk category, indicating potential dissatisfaction or declining engagement. "
        "While the customer is not at immediate risk, proactive measures such as targeted offers, improved support, or personalized "
        "communication strategies can help increase retention and prevent future churn."
    )

    else:
        conclusion = (
        "The customer exhibits a high probability of churn, which indicates a strong likelihood of disengagement or dissatisfaction. "
        "Immediate retention strategies should be implemented, such as offering discounts, resolving service issues, or direct customer outreach. "
        "Timely intervention is crucial to reduce churn and retain customer value."
    )

    t = time.time()
    return render_template(
    'index.html',
    prediction_text="done",
    churn_prob=churn_prob,
    clv=clv,
    url_1=gauge_url,
    url_2=imp_url,
    url_3=contrib_url,
    url_4=prob_url,
    url_5=value_url,
    conclusion=conclusion
)


if __name__ == "__main__":
    app.run(debug=True)
