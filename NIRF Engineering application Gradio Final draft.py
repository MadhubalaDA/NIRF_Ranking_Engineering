#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data from Excel file
FRU=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=0)
PU_QP=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx",sheet_name=1)
FPPP=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx",sheet_name=2)
GMS=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=3)
GPHD=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=4)
ESCS=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=5)
PCS=pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=6)

# Split the data into input and output
y=FRU[['FRU']]
x=FRU[['cap_avg', 'opr_avg']]
y11=PU_QP[['PU']]
x11=PU_QP[['Publications','faculty_2018']]
y12=PU_QP[['QP']]
x12=PU_QP[['Publications','Citations','Top25','faculty_2018']]
y2=FPPP[['FPPP']]
x2=FPPP[['Research','Consultancy']]
y3=GMS[['GMS']]
x3=GMS[['Salary']]
y4=GPHD[['GPHD']]
x4=GPHD[['FT_grad']]
y5=ESCS[['ESCS']]
x5=ESCS[['Reimbursed','Socially_challenged']]
y6=PCS[['PCS']]
x6=PCS[['A','B','C']]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x11_train, x11_test, y11_train, y11_test = train_test_split(x11,y11, test_size=0.2,random_state=31)
x12_train, x12_test, y12_train, y12_test = train_test_split(x12,y12, test_size=0.2,random_state=31)
x2_train, x2_test, y2_train, y2_test=train_test_split(x2,y2, test_size=0.2,random_state=31)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size=0.2,random_state=31)
x4_train, x4_test, y4_train, y4_test=train_test_split(x4,y4, test_size=0.2,random_state=31)
x5_train, x5_test, y5_train, y5_test=train_test_split(x5,y5, test_size=0.2,random_state=31)
x6_train, x6_test, y6_train, y6_test=train_test_split(x6,y6, test_size=0.2,random_state=31)


# Train the random forest regression model for SS prediction
fru_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fru_rf = fru_model.fit(x_train, y_train.values.ravel())
pu_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pu_rf = pu_model.fit(x11_train, y11_train.values.ravel())
qp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
qp_rf = qp_model.fit(x12_train, y12_train.values.ravel())
fppp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fppp_rf = fppp_model.fit(x2_train, y2_train.values.ravel())
gms_model = RandomForestRegressor(n_estimators=1000, random_state=31)
gms_rf = gms_model.fit(x3_train, y3_train.values.ravel())
gphd_model = RandomForestRegressor(n_estimators=1000, random_state=31)
gphd_rf = gphd_model.fit(x4_train, y4_train.values.ravel())
escs_model = RandomForestRegressor(n_estimators=1000, random_state=31)
escs_rf = escs_model.fit(x5_train, y5_train.values.ravel())
pcs_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pcs_rf = pcs_model.fit(x6_train, y6_train.values.ravel())

def predict_SS(SI, TE, ft):
    students = max(SI,TE)
    Ratio = min(TE/SI,1)
    if students >=10000:
        criteria = 15
    elif students <=9999 and students >=5000:
        criteria = 13.5
    elif students <=4999 and students >=4000:
        criteria = 12
    elif students <=3999 and students >=3000:
        criteria = 10.5
    elif students <=2999 and students >=2000:
        criteria = 9
    elif students <=1999 and students >=1000:
        criteria = 7.5
    elif students <=999 and students >=500:
        criteria = 6
    elif students <=499:
        criteria = 4.5
        
    student_score = round(Ratio*criteria,2)
    if ft>=1000:
        PhD_score = 5
    elif ft<=999 and ft>=500:
        PhD_score = 4
    elif ft<=499 and ft>=250:
        PhD_score = 3
    elif ft<=249 and ft>=100:
        PhD_score = 2
    elif ft<=99 and ft>=50:
        PhD_score = 1
    elif ft<=49 and ft>=25:
        PhD_score = 0.5
    else:
        PhD_score = 0
    SS = student_score + PhD_score
    return SS

def predict_FSR(Nfaculty, SI, ft, pt):
    PhD = ft + pt
    N = SI + PhD
    ratio = Nfaculty / N
    FSR = round(30 * (15 * (Nfaculty / N)),2)
    if FSR > 30:
        FSR = 30
    else:
        FSR = FSR
    if ratio < 0.02:
        FSR = 0
    round(FSR,2)
    return FSR

def predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3):
    calc_FSR=SI/15
    Faculty = max(calc_FSR,Nfaculty)
    FRA=(phd/Faculty)
    if FRA<0.95:
        FQ=10*(FRA/0.95)
    else:
        FQ=10
    F1=ft_exp1/Faculty
    F2=ft_exp2/Faculty
    F3=ft_exp3/Faculty
    FE_cal=(3*min((3*F1),1))+(3*min((3*F2),1))+(4*min((3*F3),1))
    if F1==F2==F3:
        FE=10
    else:
        FE = FE_cal
    FQE=round(FQ+FE,2)
    round(FQE,2)
    return FQE

def predict_FRU(cap_avg, opr_avg):
    # Make a prediction using the trained model for SS
    FRU = round(fru_model.predict([[cap_avg, opr_avg]])[0],2)
    return FRU

def predict_PU(Publications, faculty_2018):
    # Make a prediction using the trained model for SS
    PU = round(pu_model.predict([[Publications, faculty_2018]])[0],2)
    return PU

def predict_QP(Publications,Citations,Top25,faculty_2018):
    # Make a prediction using the trained model for SS
    QP = round(qp_model.predict([[Publications,Citations,Top25,faculty_2018]])[0],2)
    return QP

def predict_IPR(PG,PP):
    if PG >=75:
        PG_score = 10
    elif PG <=74 and PG >=50:
        PG_score = 8
    elif PG <=49 and PG >=25:
        PG_score = 6
    elif PG <=24 and PG >=10:
        PG_score = 4
    elif PG <=9 and PG >=1:
        PG_score = 2
    elif PG == 0:
        PG_score = 0
    
    if PP >=300:
        PP_score = 5
    elif PP <=299 and PP >=200:
        PP_score = 4
    elif PP <=199 and PP >=100:
        PP_score = 3
    elif PP <=99 and PP >=25:
        PP_score = 2
    elif PP <=24 and PP >=10:
        PP_score = 1
    elif PP <=9 and PP >=1:
        PP_score = 0.5
    elif PP == 0:
        PP_score = 0
    
    IPR = PG_score + PP_score
    return IPR

def predict_FPPP(Research,Consultancy):
    # Make a prediction using the trained model for SS
    FPPP = round(fppp_model.predict([[Research,Consultancy]])[0],2)
    return FPPP

def predict_GPH(Placement,HS,UG_sum):
    GPH=round((((Placement/UG_sum)+(HS/UG_sum))*40), 2)
    return GPH

def predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3):
    year1 = graduated1/si1
    year2 = graduated2/si2
    year3 = graduated3/si3
    avg=(year1+year2+year3)/3
    a=avg/0.8
    GUE=round((min(a,1)*15),2)
    return GUE

def predict_GMS(Salary):
    # Make a prediction using the trained model for SS
    GMS = round(gms_model.predict([[Salary]])[0],2)
    return GMS

def predict_GPHD(FT_grad):
    # Make a prediction using the trained model for SS
    GPHD = round(gphd_model.predict([[FT_grad]])[0],2)
    return GPHD

def predict_RD(SI,TE,other_state,other_country):
    students = max(SI,TE)
    state = (other_state/students)*25
    country = (other_country/students)*5
    RD = round((state + country),2)
    return RD

def predict_WD(WS,WF,SI,Nfaculty):
    # Make a prediction using the trained model for SS
    calc_FSR=SI/15
    Faculty = max(calc_FSR,Nfaculty)
    student_ratio=WS/SI
    faculty_ratio=WF/Faculty
    a1=min(((student_ratio)/0.5),1)
    b1=min(((faculty_ratio)/0.2),1)
    WD=round(((15*a1)+(15*b1)),2)
    return WD

def predict_ESCS(Reimbursed,Socially_challenged):
    # Make a prediction using the trained model for SS
    ESCS = round(escs_model.predict([[Reimbursed,Socially_challenged]])[0],2)
    return ESCS

def predict_PCS(A,B,C):
    # Make a prediction using the trained model for SS
    PCS = round(pcs_model.predict([[A,B,C]])[0],2)
    return PCS

def predict_all(y11,y12,y13,y14,y21,y22,y23,y24,y25,y31,y32,y41,y42,y43,y51,y52,y53,y54,y55,y56,y61,y62,y63,y64,y65, TE1,TE2,TE3,TE4,TE5,TE6, ft, pt, Nfaculty, ft_exp1, ft_exp2, ft_exp3, phd, L1,L2,L3,Lab1,Lab2,Lab3,W1,W2,W3,O1,O2,O3, S1,S2,S3,I1,I2,I3,Seminar1,Seminar2,Seminar3, faculty_2018,Publications,Citations,Top25, PG,PP,RF1,RF2,RF3,CF1,CF2,CF3, Placement11, Placement12, Placement13,Placement21,Placement22,Placement23, HS11, HS12, HS13,HS21,HS22,HS23, graduated11, graduated12, graduated13, graduated21, graduated22, graduated23, graduated31, graduated32, graduated33, graduated41, graduated42, graduated43, graduated51, graduated52, graduated53, graduated61, graduated62, graduated63,si11,si12,si13,si21,si22,si23,si31,si32,si33,si41,si42,si43,si51,si52,si53,si61,si62,si63, Salary1, Salary2, Salary3, FT_grad1, FT_grad2, FT_grad3, state1, state2, state3, state4, state5, state6, country1, country2, country3, country4, country5, country6, ws1,ws2,ws3,ws4,ws5,ws6,WF, Reimbursed11, Reimbursed12, Reimbursed13,Reimbursed21, Reimbursed22, Reimbursed23,Reimbursed31, Reimbursed32, Reimbursed33,Reimbursed41,Reimbursed42,Reimbursed43,Reimbursed51,Reimbursed52,Reimbursed53,Reimbursed61,Reimbursed62,Reimbursed63,social1,social2,social3,social4,social5,social6,economic1,economic2,economic3,economic4,economic5,economic6, A,B,C, pr):
    SI = round((y11+y12+y13+y14+y21+y22+y23+y24+y25+y31+y32+y41+y42+y43+y51+y52+y53+y54+y55+y56+y61+y62+y63+y64+y65),2)
    TE = round((TE1+TE2+TE3+TE4+TE5+TE6),2)
    ss = predict_SS(SI, TE, ft)
    fsr = predict_FSR(Nfaculty, pt, SI, ft)
    fqe = predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3)
    cap_year1 = (L1+Lab1+W1+O1)/SI
    cap_year2 = (L2+Lab2+W2+O2)/SI
    cap_year3 = (L3+Lab3+W3+O3)/SI
    cap_avg = (cap_year1+cap_year2+cap_year3)/3
    opr_year1 = (S1+I1+Seminar1)/SI
    opr_year2 = (S2+I2+Seminar2)/SI
    opr_year3 = (S3+I3+Seminar3)/SI
    opr_avg = (opr_year1+opr_year2+opr_year3)/3
    fru = predict_FRU(cap_avg, opr_avg)
    tlr = round(ss + fsr + fqe + fru, 2)
    pu = predict_PU(Publications, faculty_2018)
    qp = predict_QP(Publications,Citations,Top25,faculty_2018)
    ipr = predict_IPR(PG,PP)
    Research = ((RF1/Nfaculty)+(RF2/Nfaculty)+(RF3/Nfaculty))/3
    Consultancy = ((CF1/Nfaculty)+(CF2/Nfaculty)+(CF3/Nfaculty))/3
    fppp = predict_FPPP(Research,Consultancy)
    rp = round((pu + qp + ipr + fppp),2)
    Placement = (Placement11 + Placement12 + Placement13 + Placement21 + Placement22 + Placement23)
    HS = (HS11 + HS12 + HS13 + HS21 + HS22 + HS23)
    UG_sum = si11+si12+si13+si21+si22+si23
    gph = predict_GPH(Placement,HS,UG_sum)
    #graduated = (graduated11+ graduated12+ graduated13+ graduated21+ graduated22+ graduated23+ graduated31+ graduated32+ graduated33+ graduated41+ graduated42+ graduated43+ graduated51+ graduated52+ graduated53+ graduated61+ graduated62+ graduated63)/3
    graduated1 = (graduated11+graduated21+graduated31+graduated41+graduated51+graduated61)
    graduated2 = (graduated12+graduated22+graduated32+graduated42+graduated52+graduated62)
    graduated3 = (graduated13+graduated23+graduated33+graduated43+graduated53+graduated63)
    si1 = (si11+si21+si31+si41+si51+si61)
    si2 = (si12+si22+si32+si42+si52+si62)
    si3 = (si13+si23+si33+si43+si53+si63)
    gue = predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3)
    Salary = Salary1+ Salary2+ Salary3
    gms = predict_GMS(Salary)
    FT_grad = (FT_grad1+ FT_grad2+ FT_grad3)/3
    gphd = predict_GPHD(FT_grad)
    go = round((gph + gue + gms + gphd),2)
    other_state = (state1+ state2+ state3+ state4+ state5+ state6)
    other_country = (country1+ country2+ country3+ country4+ country5+ country6)
    rd = predict_RD(SI,TE,other_state,other_country)
    WS = ws1+ws2+ws3+ws4+ws5+ws6
    wd = predict_WD(WS,WF,SI,Nfaculty)
    Reimbursed = Reimbursed11+ Reimbursed12+ Reimbursed13+Reimbursed21+ Reimbursed22+ Reimbursed23+Reimbursed31+ Reimbursed32+ Reimbursed33+Reimbursed41+Reimbursed42+Reimbursed43+Reimbursed51+Reimbursed52+Reimbursed53+Reimbursed61+Reimbursed62+Reimbursed63
    Socially_challenged = social1+social2+social3+social4+social5+social6+economic1+economic2+economic3+economic4+economic5+economic6
    escs = predict_ESCS(Reimbursed,Socially_challenged)
    pcs = predict_PCS(A,B,C)
    oi = round((rd + wd + escs + pcs),2)
    Overall_score = round(((tlr * 0.3) + (rp * 0.3) + (go * 0.2) + (oi * 0.1) + (pr * 0.1)),2)
    
    return (ss, fsr, fqe, fru, tlr, pu, qp, ipr, fppp, rp, gph, gue, gms, gphd, go, rd, wd, escs, pcs, oi, Overall_score)

# Create a Gradio interface
inputs = [
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 2 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 2 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year6", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year5", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG integrated", default=60),
    gr.inputs.Number(label="Number of PhD Enrolled Full Time", default=10),
    gr.inputs.Number(label="Number of PhD Enrolled Part Time", default=5),
    gr.inputs.Number(label="No. of Full Time Regular Faculty", default=20),
    gr.inputs.Number(label="No. of full time regular faculty with Experience up to 8 years", default=5),
    gr.inputs.Number(label="No. of full time regular faculty with Experience between 8+ to 15 years", default=10),
    gr.inputs.Number(label="No. of full time regular faculty with Experience > 15 years", default=15),
    gr.inputs.Number(label="No. of faculty with PhD", default=5),
    gr.inputs.Number(label="Annual Expenditure on Library-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Library-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Library-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year3", default=60),
    gr.inputs.Number(label="No. of Full Time Regular Faculty", default=20),
    gr.inputs.Number(label="No. of Publications", default=60),
    gr.inputs.Number(label="No. of citations", default=60),
    gr.inputs.Number(label="No. of Top25 percentage", default=60),
    gr.inputs.Number(label="No. of Patent_Granted", default=60),
    gr.inputs.Number(label="No. of Patent_Published", default=60),
    gr.inputs.Number(label="Amount received in sponsored research - year1", default=60),
    gr.inputs.Number(label="Amount received in sponsored research - year2", default=60),
    gr.inputs.Number(label="Amount received in sponsored research - year3", default=60),
    gr.inputs.Number(label="Amount received in consultancy projects - year1", default=60),
    gr.inputs.Number(label="Amount received in consultancy projects - year2", default=60),
    gr.inputs.Number(label="Amount received in consultancy projects - year3", default=60),
    gr.inputs.Number(label="No. of students placed(UG 4 year) - year1", default=60),
    gr.inputs.Number(label="No. of students placed(UG 4 year) - year2", default=60),
    gr.inputs.Number(label="No. of students placed(UG 4 year) - year3", default=60),
    gr.inputs.Number(label="No. of students placed(UG 5 year) - year1", default=60),
    gr.inputs.Number(label="No. of students placed(UG 5 year) - year2", default=60),
    gr.inputs.Number(label="No. of students placed(UG 5 year) - year3", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 4 year) - year1", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 4 year) - year2", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 4 year) - year3", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 5 year) - year1", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 5 year) - year2", default=60),
    gr.inputs.Number(label="No. of students selected for HS(UG 5 year) - year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year3", default=60),
    gr.inputs.Number(label="Total Median Salary - year1", default=60),
    gr.inputs.Number(label="Total Median Salary - year2", default=60),
    gr.inputs.Number(label="Total Median Salary - year3", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year1", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year2", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year3", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG integrated", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG integrated", default=60),
    gr.inputs.Number(label="No. of women students - UG 4 year", default=60),
    gr.inputs.Number(label="No. of women students - UG 5 year", default=60),
    gr.inputs.Number(label="No. of women students - PG 2 year", default=60),
    gr.inputs.Number(label="No. of women students - PG 3 year", default=60),
    gr.inputs.Number(label="No. of women students - PG 6 year", default=60),
    gr.inputs.Number(label="No. of women students - PG integrated", default=60),
    gr.inputs.Number(label="No. of women faculty", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Government - PG integrated", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from Institute - PG integrated", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from private bodies - PG integrated", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG integrated", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG integrated", default=60),
    gr.inputs.Number(label="Lifts/Ramps", default=60),
    gr.inputs.Dropdown(choices = ["1", "0"],label="Walking aids", default=1),
    gr.inputs.Number(label="Specially designed toilets for handicapped students", default=60),
    gr.inputs.Number(label="PR", default=60)
]
output1 = gr.outputs.Textbox(label="SS")
output2 = gr.outputs.Textbox(label="FSR")
output3 = gr.outputs.Textbox(label="FQE")
output4 = gr.outputs.Textbox(label="FRU")
output_tlr = gr.outputs.Textbox(label="TLR")
output5 = gr.outputs.Textbox(label="PU")
output6 = gr.outputs.Textbox(label="QP")
output7 = gr.outputs.Textbox(label="IPR")
output8 = gr.outputs.Textbox(label="FPPP")
output_rp = gr.outputs.Textbox(label="RP")
output9 = gr.outputs.Textbox(label="GPH")
output10 = gr.outputs.Textbox(label="GUE")
output11 = gr.outputs.Textbox(label="GMS")
output12 = gr.outputs.Textbox(label="GPHD")
output_go = gr.outputs.Textbox(label="GO")
output13 = gr.outputs.Textbox(label="RD")
output14 = gr.outputs.Textbox(label="WD")
output15 = gr.outputs.Textbox(label="ESCS")
output16 = gr.outputs.Textbox(label="PCS")
output_oi = gr.outputs.Textbox(label="OI")
output_score = gr.outputs.Textbox(label="Overall_score")

gradio_interface = gr.Interface(fn=predict_all, inputs=inputs, outputs=[output1, output2, output3, output4, output_tlr, output5, output6, output7, output8, output_rp, output9, output10, output11, output12, output_go, output13, output14, output15, output16, output_oi, output_score], title="Engineering NIRF Score calculation", 
                                description="Enter the input parameters to predict Overall score")

# Deploy your Gradio interface to Gradio's hosting service
gradio_interface.launch(share=True)


# In[ ]:





# In[ ]:




