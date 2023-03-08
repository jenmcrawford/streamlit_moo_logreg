import numpy as np
import matplotlib.pyplot as plt
from sklearn import LogisticRegression

##############################################################################
#
# Use plotting functions from initial notebook
#
#############################################################################

def plot_fit_1D(feat, df_combined, save_fig=False):  
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

def heatmap_logreg(feat1, feat2, df_train, df_test, model, annotate_test = False, annotate_train = False):  
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
    
    plt.figure(figsize=(9.5,8))
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
    plt.show()

