import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,RepeatedKFold,cross_val_score,cross_validate

from sigman_logreg.logreg_stats import calc_mcfad, calc_mcfadden_R2, precision_recall_f1_score, test_accuracy_score, kfold_logreg 

import streamlit as st

##############################################################################
#
# Use plotting functions from initial notebook - added global variables as function arguments
#
#############################################################################

def plot_fit_1D(feat, df_combined, X_labelname_dict,save_fig=False):  
    X_train = np.array(df_combined.loc[:, [feat]])
    y_train = np.array(df_combined.iloc[:, -1])
    lr = LogisticRegression().fit(X_train,y_train)
    m, b = lr.coef_[0][0], lr.intercept_[0]
    feat_name = X_labelname_dict[feat] #where is this variable defined

    x_min, x_max = min(X_train), max(X_train)
    x_range = x_max - x_min
    plot_min, plot_max = float(x_min-0.05*x_range), float(x_max+0.05*x_range)

    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=26) 
    plt.yticks(fontsize=26)
    plt.xlabel(feat_name,fontsize=26)
    plt.ylabel('Probability',fontsize=26)
    #plt.locator_params(axis='y', nbins=4)
    #plt.locator_params(axis='x', nbins=5)
    plt.axvline(x=(-b/m),alpha=1,c='black')
    plt.axvline(x=((np.log(3)-b)/m),alpha=1,c='black', linestyle = '--')
    plt.axvline(x=((np.log(1/3)-b)/m),alpha=1,c='black', linestyle = '--')
    plt.scatter(X_train, y_train, label="training", alpha=1,marker='o', c ='Blue', s=150  ,edgecolor='black')
    plt.xlim(plot_min, plot_max)
    plt.ylim(-.05, 1.05)

    x = np.linspace(x_min-0.4*x_range, x_max+0.4*x_range)
    f_x = np.exp(b + m*x)
    y_sigmoid = f_x/(f_x + 1)
    plt.plot(x, y_sigmoid, color = 'black')
    if save_fig:
        plt.savefig(feat + '.png', dpi=500)
    plt.show()

def heatmap_logreg(feat1, feat2, df_train, df_test, model, X_names,y_train,y_labels, TS, VS,colormap_scheme='PiYG', point_color='Greys', annotate_test = False, annotate_train = False):  
    #Train model on features
    p1_vals = df_train.iloc[:, feat1]
    p2_vals = df_train.iloc[:, feat2]
    p1_vals_test = df_test.iloc[:, feat1]
    p2_vals_test = df_test.iloc[:, feat2]
    lr = LogisticRegression().fit(df_train.iloc[:,[feat1, feat2]],y_train)
    intercept = lr.intercept_[0]
    c1, c2 = lr.coef_[0][0], lr.coef_[0][1]
    p1, p2 = X_names[feat1], X_names[feat2]
    
    #Get max/min values for X and Y axes
    max_x, min_x = max(list(p1_vals) + list(p1_vals_test)), min(list(p1_vals) + list(p1_vals_test))
    max_y, min_y = max(list(p2_vals) + list(p2_vals_test)), min(list(p2_vals) + list(p2_vals_test))
    range_x, range_y = abs(max_x - min_x), abs(max_y - min_y)
    max_x_plt, min_x_plt = max_x + 0.1*range_x, min_x - 0.1*range_x
    max_y_plt, min_y_plt = max_y + 0.1*range_y, min_y - 0.1*range_y
    
    #heatmap code
    xx = np.linspace(min_x_plt, max_x_plt, 500)
    yy = np.linspace(min_y_plt, max_y_plt, 500)
    xx,yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
   
    # Predict probabilities of full grid
    probas = lr.predict_proba(Xfull)
    
    fig = plt.figure(figsize=(9.5,8))
    plt.xticks(fontsize=25, fontname = 'Helvetica') 
    plt.yticks(fontsize=25, fontname = 'Helvetica')
    plt.xlabel(p1,fontsize=25, fontname = 'Helvetica',labelpad=20)
    plt.ylabel(p2,fontsize=25, fontname = 'Helvetica', labelpad=20)
    
    heatmap = plt.imshow(
            probas[:,1].reshape((500, 500)), cmap=colormap_scheme, alpha = 0.5, extent=[min_x_plt, max_x_plt, min_y_plt, max_y_plt],interpolation='nearest', origin="lower"
        ,aspect='auto')
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Probability', size=25, fontname = 'Helvetica', labelpad=20)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.scatter(p1_vals, p2_vals, label="training", alpha=1,marker='o', c = df_train.iloc[:, -1], cmap=point_color + '_r', s=200  ,edgecolor='black')
    plt.scatter(p1_vals_test, p2_vals_test, label="test", alpha=1,marker='X' , c = df_test.iloc[:, -1], cmap=point_color + '_r', s=200  ,edgecolor='black')
    #plt.colorbar()
    #cbar.set_label('yield %',rotation=90,size=22,labelpad=20)

    x = np.linspace(min_x - 0.1*range_x, max_x + 0.1*range_x)
    y = -(intercept/c2) - (c1/c2)*x
    y_75 = ((np.log(3)-intercept)/c2) - (c1/c2)*x    #
    y_25 = ((np.log(1/3)-intercept)/c2) - (c1/c2)*x
    plt.xlim([min_x - 0.1*range_x, max_x + 0.1*range_x])
    plt.ylim([min_y - 0.1*range_y, max_y + 0.1*range_y])

    plt.plot(x, y, color = 'black')
    plt.plot(x,y_75, color = 'black', linestyle = '--')
    plt.plot(x,y_25, color = 'black', linestyle = '--')
    
    if annotate_test == True:
        for i, txt in enumerate(VS):
            label_i = y_labels[txt]
            plt.annotate(label_i, (p1_vals_test[i], p2_vals_test[i]), fontsize='14', c = 'white')
    if annotate_train == True:
        for i, txt in enumerate(TS):
            label_i = y_labels[txt]
            plt.annotate(label_i, (p1_vals[i], p2_vals[i]), fontsize='14', c = 'white')
    plt.tight_layout()
    plt.savefig('Logistic_Regression.png', dpi=500)
    st.pyplot(fig)

def data_prep_fun(X,X_sel,y_sel,split="random",test_ratio=0.2):
    if split == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y_sel, random_state=42, test_size=test_ratio)    
        TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train]
        VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test]

        return X_train, X_test, y_train, y_test, TS, VS

    #removed other options define and none b/c of variable situation

    else: 
        raise ValueError("split option not recognized")

def logreg_prep(X_train,y_train,X_test,y_test,skipfeatures=[]):
    df_train = pd.DataFrame(np.hstack((X_train,y_train[:,None])))     
    newcols = ["x"+str(i+1) for i in df_train.columns.values]
    df_train.columns = newcols
    response = newcols[-1]
    df_train.rename(columns={response:"y"},inplace=True)
    df_train.drop(skipfeatures,axis=1,inplace=True)


    df_test = pd.DataFrame(np.hstack((X_test,y_test[:,None])))
    newcols = ["x"+str(i+1) for i in df_test.columns.values]
    df_test.columns = newcols
    response = newcols[-1]
    df_test.rename(columns={response:"y"},inplace=True)
    df_test.drop(skipfeatures,axis=1,inplace=True)

    return df_train, df_test, newcols, response

def man_sel_feat(X_train,X_test,y_train,y_test,X_labels,X_labelname,X_names,y_labels,TS,VS,df_train,df_test,features_x,annotate_test_set=False,annotate_training_set=False):
    selected_feats = sorted([X_labels.index(i.strip()) for i in features_x])
    X_train_sel = X_train[:,selected_feats]
    X_test_sel = X_test[:,selected_feats]
    lr = LogisticRegression().fit(X_train_sel,y_train)
    y_pred_train = lr.predict(X_train_sel)
    if len(VS) > 0:  
        test_accuracy = test_accuracy_score(X_test_sel, y_test, X_train_sel, y_train)  
    kfold_score, kfold_stdev = kfold_logreg(X_train_sel, y_train)
    precision, recall, f1 = precision_recall_f1_score(X_train_sel, y_train)
    train_R2, test_R2 = calc_mcfadden_R2(X_train_sel, y_train, X_test_sel, y_test)

    print("Parameters and coefficients:\n{:10.4f} + \n".format(lr.intercept_[0]) + "\n".join(["{:10.4f} * {}".format(lr.coef_[0][i],X_labelname[selected_feats[i]]) for i in range(len(selected_feats))]))
    print(f"\nMcFadden Training R2  = {train_R2 :.2f}")
    if len(VS) > 0:
        print(f"McFadden Test R2  = {test_R2 :.2f}")
    print(f"\nAccuracy  = {100 * lr.score(X_train_sel, y_train):.1f} %")
    if len(VS) > 0:
        print(f"Test Accuracy  = {100 * lr.score(X_test_sel, y_test):.1f}%")
    print("\nTraining K-fold Accuracy = {:.2f} (+/- {:.2f}) %".format(100* kfold_score, 100* kfold_stdev ** 2))
    print(f"f1 Score  = {f1:.3f}")
    print(f"Precision Score  = {precision :.3f}")
    print(f"Recall Score  = {recall:.3f}")
    print('\nNote: \n(1) Black and white points denote active and inactive ligands respectively.')
    print('(2) Red and blue denote active and inactive chemical space respectively.')
    print('(3) Dashed lines denote 25% and 75% probability that a ligand will be active.')
    print('(4) Solid black line denotes 50% probability that a ligand will be active.')

    heatmap_logreg(selected_feats[0],selected_feats[1],df_train,df_test,features_x, X_names,y_train,y_labels, TS, VS,annotate_test=False, annotate_train=False)

def forward_step_logreg():
    pass
    