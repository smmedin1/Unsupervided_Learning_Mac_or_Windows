#!/usr/bin/env python
# coding: utf-8

# <span><small>Hult International Business School</small></span>
# <span style="margin-left: 17%"><small>Machine Learning, Business Analytics</small></span>
# <span style="float: right"><small>MsBA2&nbsp; Spring 2021</small></span>
# <hr style="color: black; display: block; height: 1px; border: ; margin-bottom: 1.5em; margin-top: .3em">
# <h3 style="margin-top: 2em">Team 12 Project - Unsupervised Analysis </h3>
# <center>
#     <h2 style="margin-top: 1.5em; color: #000">Business Report</h2>
#     <br>
#     <div style="margin-right: 3.5em"><table border="0">
#         <tr><td>Course:&nbsp;</td>     <td><span style="padding-right: 1.5em">DAT-5303 Machine Learning</span></td></tr>
#      <tr><td>Instructor:&nbsp;</td> <td><span style="padding-right: 2.7em">Prof. Chase B. Kusterer</span></td></tr>
#      <tr><td>Author:&nbsp;</td>     <td><span style="padding-right: 5.2em">Team 12 MsBA2</span></td></tr>
#      <tr><td>Institution:&nbsp;</td><td>Hult International Business School</td></tr>
#      <tr><td>Date:&nbsp;</td>       <td><span style="padding-right: 4em">January 31st, 2021</span></td></tr>
#      <tr><td colspan="2"></td></tr>
#      <tr><td colspan="2"></td></tr>
#      <tr><td colspan="2"><img src="./hult_icon.png" width="42px" /></td></tr>
#      <tr><td colspan="2"></td></tr>
#     </table></div>
# </center>
# </div>

# <h2> Objective </h2><br>
# Today, we will be analyzing survey data conducted by Apple and Microsoft. By researching several aspects of consumer behavior in regards to the decision behind the question "Windows or Mac?", special attention is being drawn on the five most popular personality traits among those surveyed as well as the top three relations to HULT DNA. These top features consisted of being an introvert, Outgoing, Scatter Brained, Hippie, Self-Motivated, along with whether surveyors relate to being a follower, creative, and reactive. We will go into further detail on  potential customers and their answers to the survey.

# <h2> Exploring the Data </h2><br>

# <h3> Importing Packages </h3><br>

# In[ ]:


########################################
# importing packages
########################################
import numpy             as np                   # mathematical essentials
import pandas            as pd                   # data science essentials
import matplotlib.pyplot as plt                  # fundamental data visualization
import seaborn           as sns                  # enhanced visualization
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA            # pca
from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms
from sklearn.cluster         import KMeans

########################################
# loading data and setting display options
########################################



# setting print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

# specify path and file name
file = 'survey_data.xlsx'


# reading the file
cust_df = pd.read_excel(io=file)

#pd set option to see all the column names 
pd.set_option('display.max_columns', None)
cust_df.head(n=5)


# In[ ]:


cust_df.shape


# <h3> User Defined Functions

# In[ ]:


########################################
# inertia
########################################
def interia_plot(data, max_clust = 50):
    """
PARAMETERS
----------
data      : DataFrame, data from which to build clusters. Dataset should be scaled
max_clust : int, maximum of range for how many clusters to check interia, default 50
    """

    ks = range(1, max_clust)
    inertias = []


    for k in ks:
        # INSTANTIATING a kmeans object
        model = KMeans(n_clusters = k)


        # FITTING to the data
        model.fit(data)


        # append each inertia to the list of inertias
        inertias.append(model.inertia_)



    # plotting ks vs inertias
    fig, ax = plt.subplots(figsize = (12, 8))
    plt.plot(ks, inertias, '-o')


    # labeling and displaying the plot
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


########################################
# scree_plot
########################################
def scree_plot(pca_object, export = False):
    # building a scree plot

    # setting plot size
    fig, ax = plt.subplots(figsize=(10, 8))
    features = range(pca_object.n_components_)


    # developing a scree plot
    plt.plot(features,
             pca_object.explained_variance_ratio_,
             linewidth = 2,
             marker = 'o',
             markersize = 10,
             markeredgecolor = 'black',
             markerfacecolor = 'grey')


    # setting more plot options
    plt.title('Scree Plot')
    plt.xlabel('PCA feature')
    plt.ylabel('Explained Variance')
    plt.xticks(features)

    if export == True:
    
        # exporting the plot
        plt.savefig('./analysis_images/top_customers_correlation_scree_plot.png')
        
    # displaying the plot
    plt.show()


# In[ ]:


#taking out the extra columns
cust_df.columns = [col.replace(',', '') for col in cust_df.columns]


# In[ ]:


# checking information about each column
cust_df.isnull().sum(axis=0)


# In[ ]:


# summary of decriptive statistics
cust_df.describe(include='number').round(decimals=2)


# <h2> Engineering the Data </h2><br>

# <h3> Checking Demographics </h3>

# In[ ]:


#selected the categorical data (Demographics) and analyzed the value counts. 
# value counts for Survery Id
print(cust_df['surveyID'].value_counts())


print("\n\n")


# value counts for Age
print(cust_df['What is your age?'].value_counts())


# value counts for Gender
print(cust_df['Gender'].value_counts())



# value counts for Laptop
print(cust_df['What laptop do you currently have?'].value_counts())


# value counts for laptop in next
print(cust_df['What laptop would you buy in next assuming if all laptops cost the same?'].value_counts())

# Value counts for ethnicity

print(cust_df['What is your ethnicity?'].value_counts())


# In[ ]:


#printing value counts for cat variables

print("""What laptop do you currently have?\n""",
cust_df['What laptop do you currently have?'].value_counts()
)

print("""\nWhat laptop would you buy in next assuming if all laptops cost the same?\n""",
cust_df['What laptop would you buy in next assuming if all laptops cost the same?'].value_counts()
)

print("""\nWhat program are you in?\n""",
cust_df['What program are you in?'].value_counts()
)

print("""\nWhat is your age?\n""",
cust_df['What is your age?'].value_counts()
)

print("""\nWhat is your ethnicity?\n""",
cust_df['What is your ethnicity?'].value_counts()
)

print("""\nWhat is your gender?\n""",
cust_df['Gender'].value_counts()
)

print("""\nWhat is your nationality?\n""",
cust_df['What is your nationality? '].value_counts()
)


# In[ ]:


#Changing to lower case

cust_df['What is your nationality? '] = cust_df['What is your nationality? '].str.lower()

cust_df['What is your nationality? '].value_counts()


# In[ ]:


#creating a map for nationalities
cust_df['What is your nationality? '] = cust_df['What is your nationality? '].map({

          'usa': 'american',
          'american': 'american',
          'belarus': 'belarus', 
          'belgium': 'belgium',
          'brazil': 'brazilian',
          'brazilian': 'brazilian',
          'british': 'british',
          'british, indian': 'multinational',
          'canada': 'canadian',
          'canadian': 'canadian', 
          'china':'chinese',
          'chinese': 'chinese',
          'colombia' : 'colombian',
          'colombian': 'colombian',
          'costarrican': 'costarrican',
          'congolese (dr congo)': 'congolese',
          'congolese': 'congolese', 
          'czech republic': 'czech',
          'czech': 'czech',
          'dominican ':'dominican' ,
          'ecuador': 'ecuadorian',
          'ecuadorian': 'ecuadorian', 
          'filipino': 'phillipine',
          'phillipines': 'phillipine', 
          'germany': 'german',
          'german': 'german',
          'german/american': 'multinational',
          'ghanian': 'ghanian',
          'indian': 'indian',
          'indian.': 'indian',
          'indonesia': 'indonesian',
          'indonesian': 'indonesian',
          'italian': 'italian',
          'italian and spanish': 'multinational',
          'japan': 'japanese',
          'japanese': 'japanese',
          'kenyan': 'kenyan',
          'republic of korea': 'korean',
          'korea': 'korean',
          'south korea': 'korean',
          'kyrgyz': 'kyrgyz',
          'mauritius': 'mauritius',  
          'mexican': 'mexican',
          'nigeria': 'nigerian',
          'nigerian': 'nigerian',
          'norwegian': 'norwegian',
          'pakistani': 'pakistani',
          'peru': 'peruvian',
          'peruvian': 'peruvian', 
          'panama': 'panamanian',
          'panamanian' : 'panamanian',
          'portuguese': 'portuguese',
          'prefer not to answer': 'prefer not to answer',
          'russia': 'russian',
          'russian': 'russian',
          'spain': 'spanish',
          'spanish': 'spanish',
          'swiss': 'swiss',
          'taiwan': 'taiwan',
          'thai' : 'thai',
          'turkish': 'turkish',
          'ugandan': 'ugandan',
          'ukrainian' : 'ukrainian',
          'venezuelan': 'venezuelan',
          'vietnamese': 'vietnamese'

        
})


# In[ ]:


#Changing to lower case

cust_df['What is your nationality? '] = cust_df['What is your nationality? '].str.lower()

cust_df['What is your nationality? '].value_counts()


# <h3>Cross-Checking Opposite/Similar Pooling Questions </h3><br>

# 
# The dataset contains similar and opposite questions. 
# 1. Similar questions: observations that presented extreme answers (1 and 5) for questions with the same topic were dropped from the survey.
# 2. Opposite questions: observations that presented different answers, meaning the same score for opposite questions, were also removed only this inconsistency occurred more than once.

# In[ ]:


#looking at column multiples
for index, row in cust_df.iterrows():
    if abs(row['Encourage direct and open discussions'] - row['Encourage direct and open discussions.1']) >= 1 and    abs(row["Take initiative even when circumstances objectives or rules aren't clear"] - row["Take initiative even when circumstances objectives or rules aren't clear.1"]) >= 1 and     abs(row['Respond effectively to multiple priorities'] - row['Respond effectively to multiple priorities.1']) >= 1:
        cust_df.drop(index, inplace = True)


cust_df.shape


# In[ ]:


#similarities
for index, row in cust_df.iterrows():
    if row['Feel little concern for others']- row['Am not really interested in others'] > abs(3):
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row['Have a rich vocabulary']- row['Use difficult words'] > abs(3): #did not change 
        cust_df.drop(index, inplace = True)        
        
for index, row in cust_df.iterrows():
    if row["Don't talk a lot"]- row['Have little to say'] > abs(3):
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row['Am interested in people']- row['Feel comfortable around people'] > abs(3): #did not change
        cust_df.drop(index, inplace = True)

for index, row in cust_df.iterrows():
    if row['Leave my belongings around']- row['Often forget to put things back in their proper place'] > abs(3): 
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row['Seldom feel blue']- row['Often feel blue'] > abs(3): 
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row['Get irritated easily']- row['Am easily disturbed'] > abs(3): 
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row["Sympathize with others' feelings"]- row["Feel others' emotions"] > abs(3): 
        cust_df.drop(index, inplace = True)

        
for index, row in cust_df.iterrows():
    if row['Am the life of the party']- row['Talk to a lot of different people at parties'] > abs(3): #did not change
        cust_df.drop(index, inplace = True)

for index, row in cust_df.iterrows():
    if row['Change my mood a lot']- row['Have frequent mood swings'] > abs(3): #did not change
        cust_df.drop(index, inplace = True)
        
for index, row in cust_df.iterrows():
    if row['Am not interested in abstract ideas']- row['Have difficulty understanding abstract ideas'] > abs(3): 
        cust_df.drop(index, inplace = True) #did not change


cust_df.shape


# In[ ]:


#polar opposites
for index, row in cust_df.iterrows():
    if row['Have a vivid imagination']- row['Do not have a good imagination'] == 0 and     row['Am interested in people']- row['Am not really interested in others'] == 0 and     row['Am full of ideas']- row["Don't  generate ideas that are new and different"] == 0 and     row["Don't mind being the center of attention"]- row["Don't like to draw attention to myself"] == 0 and     row["Don't talk a lot"]- row['Talk to a lot of different people at parties'] == 0:
        cust_df.drop(index, inplace = True)

        
cust_df.shape


# In[ ]:


# value counts for channel
print(cust_df['surveyID'].value_counts())



# In[ ]:


print(cust_df['What is your age?'].value_counts())


# In[ ]:


#creating a categorical dataframe
categ_df= cust_df.copy()


# In[ ]:


#dropping noncategorical varoiables
categ_df= categ_df.drop(['Am the life of the party', 'Feel little concern for others', 'Am always prepared', 
                         'Get stressed out easily', 'Have a rich vocabulary', "Don't talk a lot", 
                         'Am interested in people', 'Leave my belongings around', 
                         'Am relaxed most of the time', 'Have difficulty understanding abstract ideas', 
                         'Feel comfortable around people', 'Insult people', 'Pay attention to details', 
                         'Worry about things', 'Have a vivid imagination', 'Keep in the background', 
                         "Sympathize with others' feelings", 'Make a mess of things', 'Seldom feel blue', 
                         'Am not interested in abstract ideas', 'Start conversations', 
                         "Am not interested in other people's problems", 'Get chores done right away', 
                         'Am easily disturbed', 'Have excellent ideas', 'Have little to say', 
                         'Have a soft heart', 'Often forget to put things back in their proper place', 
                         'Get upset easily', 'Do not have a good imagination', 
                         'Talk to a lot of different people at parties', 'Am not really interested in others', 
                         'Like order',
       'Change my mood a lot', 'Am quick to understand things', "Don't like to draw attention to myself", 
                         'Take time out for others', 'Shirk my duties', 'Have frequent mood swings', 
                         'Use difficult words', "Don't mind being the center of attention", 
                         "Feel others' emotions", 'Follow a schedule', 'Get irritated easily', 
                         'Spend time reflecting on things', 'Am quiet around strangers', 
                         'Make people feel at ease', 'Am exacting in my work', 'Often feel blue', 
                         'Am full of ideas', 'See underlying patterns in complex situations', 
                         "Don't  generate ideas that are new and different", 
                         'Demonstrate an awareness of personal strengths and limitations', 
                         'Display a growth mindset', 'Respond effectively to multiple priorities', 
                         "Take initiative even when circumstances objectives or rules aren't clear", 
                         'Encourage direct and open discussions', 
                         'Respond effectively to multiple priorities.1', 
                         "Take initiative even when circumstances objectives or rules aren't clear.1",
       'Encourage direct and open discussions.1', 'Listen carefully to others', 
                         "Don't persuasively sell a vision or idea", 'Build cooperative relationships', 
                         'Work well with people from diverse cultural backgrounds', 
                         'Effectively negotiate interests resources and roles', 
                         "Can't rally people on the team around a common goal", 
                         'Translate ideas into plans that are organized and realistic', 
                         'Resolve conflicts constructively', 'Seek and use feedback from teammates', 
                         'Coach teammates for performance and growth', 'Drive for results'], axis=1)


# <h2> Customer Personality Traits PCA </h2>

# In[ ]:


#cust new dataframe
cust_df_new= cust_df.copy()
cust_df_new= cust_df_new.drop(['What laptop do you currently have?', 
                            'What laptop would you buy in next assuming if all laptops cost the same?',
                            'What program are you in?', 
                            'What is your age?', 
                            'What is your ethnicity?', 
                            'Gender', 
                            'What is your nationality? ', 
                            'surveyID','See underlying patterns in complex situations', 
                            'Demonstrate an awareness of personal strengths and limitations',                
                            'Display a growth mindset',                                                      
                            'Respond effectively to multiple priorities',                                    
                            'Listen carefully to others',                                                                                          
                            'Build cooperative relationships',                                               
                            'Work well with people from diverse cultural backgrounds',                       
                            'Effectively negotiate interests resources and roles',                                                     
                            'Translate ideas into plans that are organized and realistic',                   
                            'Resolve conflicts constructively',                                              
                            'Seek and use feedback from teammates',                                          
                            'Coach teammates for performance and growth','Am full of ideas'],
                     axis=1)


# In[ ]:


#Scaling the data

scaler = StandardScaler()
scaler.fit(cust_df_new)
X_scaled = scaler.transform(cust_df_new)

#Converting scaled data into a dataframe
cust_scaled = pd.DataFrame(X_scaled)

#Reattaching column names
cust_scaled.columns = cust_df_new.columns

#Checking pre- and post scaling variance
print(pd.np.var(cust_df_new), '\n\n')
print(pd.np.var(cust_scaled))


# In[ ]:


# INSTANTIATING a PCA object with no limit to principal components
pca = PCA(n_components = None,
            random_state = 802)


# FITTING and TRANSFORMING the scaled data
cust_pca = pca.fit_transform(cust_scaled)


# comparing dimensions of each DataFrame
print("Original shape:", cust_scaled.shape)
print("PCA shape     :",  cust_pca.shape)


# <h3>Principal Component for Personality</h3>
#  

# In[ ]:


#calling the Dataframe
pd.DataFrame(cust_pca)


# In[ ]:


# component number counter
component_number = 0
sum=0
# looping over each principal component
for variance in pca.explained_variance_ratio_:
    component_number += 1
    sum=sum+variance
    print(f"PC {component_number}    : {variance.round(3)}    : {sum.round(2)}")


# In[ ]:


# printing the sum of all explained variance ratios
print(pca.explained_variance_ratio_.sum())


# <h3> Principal Component Analysis </h3>

# Our process here is to:
# 1. Develop a PCA model with no limit to principal components
# 2. Analyze the explained_variance_ratio and the scree plot
# 3. Decide how many components to RETAIN
# 4. Build a new model with a limited number of principal components
# 5. Interpret the results (what does each PC represent)

# In[ ]:


# INSTANTIATING a PCA object with no limit to principal components
pca = PCA(n_components = None,
          random_state = 219)


# FITTING and TRANSFORMING the purchases_scaled
cust_pca = pca.fit_transform(cust_scaled)


# calling the scree_plot function
scree_plot(pca_object = pca)


# In[ ]:


#Splitting for Hult (Personality traits???)
cust_s = cust_scaled.iloc[:, 0:23]
cust_s.head()

#Instantiating PCA object with no limit to principal components for BIG 5
cust_pca = PCA(n_components = None,
         random_state = 802)

#Fitting and transforming
cust_pca_fit = cust_pca.fit_transform(cust_s)

#Calling scree plot function
scree_plot(pca_object = cust_pca)

# I am choosing 3 features


# In[ ]:


#PCA for BIG 5
pca_5 = PCA(n_components = 5,
           random_state = 802)

#Fitting and Transforming 
cust_pca_5 = pca_5.fit_transform(cust_s)

#Calling the scree_plot function
scree_plot(pca_object = pca_5)


# In[ ]:


#PCA for personalities
pca_3 = PCA(n_components = 3,
           random_state = 802)

#Fitting and Transforming 
cust_pca_3 = pca_3.fit_transform(cust_s)

#Calling the scree_plot function
scree_plot(pca_object = pca_3)


# <h3> Personality Correlation Heatmap

# In[ ]:


# setting plot size
fig, ax = plt.subplots(figsize = (15, 10))


# developing a PC to feature heatmap
sns.heatmap(pca_5.components_, 
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linewidths = 0.1,
            linecolor = 'black')




plt.xlabel(xlabel = "Feature")
plt.ylabel(ylabel = "Principal Component")


# displaying the plot
plt.show()


# <h3> Factor Loading- Personalities

# In[ ]:


#Max PC Model for BIG 5

# transposing pca components (pc = MAX)
factor_loadings = pd.DataFrame(pd.np.transpose(cust_pca.components_))


# naming rows as original features
factor_loadings = factor_loadings.set_index(cust_s.columns)


##################
### 5 PC Model ###
##################
# transposing pca components (pc = 5)
factor_loadings_5 = pd.DataFrame(pd.np.transpose(pca_5.components_))


# naming rows as original features
factor_loadings_5 = factor_loadings_5.set_index(cust_s.columns)


# checking the results
print(f"""
MAX Components Factor Loadings
------------------------------
{factor_loadings.round(2)}


5 Components Factor Loadings
------------------------------
{factor_loadings_5.round(2)}
""")


# In[ ]:


#factor loadings big 5
factor_loadings_5.columns= ['Introvert', 'Outgoing', 'Scatter Brained', 'Hippie', 'Narcissist']
factor_loadings_5


# In[ ]:


# analyzing factor strengths per customer
X_pca_reduced = pca_5.transform(cust_s)


# converting to a DataFrame
X_pca_df = pd.DataFrame(X_pca_reduced)


# renaming columns
X_pca_df.columns = factor_loadings_5.columns


# checking the results
X_pca_df


# The likelihood of someone purchasing a computer based on these top five personalities was evaluated through standard deviation. Extroverts seem to have more interest and intrigue in purchasing habits. Whether a customer is outgoing or not, does not affect behaviour. Hippies are more relaxed and tend to be less stressed would rather buy a Mac. People who are less selfish buy more macs than windows.  This insight may also say that stressed out college students or busy entrepreneurs probably buy more Apple products. 

# In[ ]:


#Introvert
len(X_pca_df['Introvert'][X_pca_df['Introvert'] > 0.5])/len(X_pca_df)


# In[ ]:


len(X_pca_df['Introvert'][X_pca_df['Introvert'] < -0.5])/len(X_pca_df)
#these customers are probably more extroverted


# In[ ]:


#Outgoing
len(X_pca_df['Outgoing'][X_pca_df['Outgoing'] > 0.5])/len(X_pca_df)


# In[ ]:


len(X_pca_df['Outgoing'][X_pca_df['Outgoing'] < -0.5])/len(X_pca_df)


# In[ ]:


#scatter brained
len(X_pca_df['Scatter Brained'][X_pca_df['Scatter Brained'] > 0.5])/len(X_pca_df)


# In[ ]:


len(X_pca_df['Scatter Brained'][X_pca_df['Scatter Brained'] < -0.5])/len(X_pca_df)


# In[ ]:


#Hippie
len(X_pca_df['Hippie'][X_pca_df['Hippie'] > 0.5])/len(X_pca_df)


# In[ ]:


len(X_pca_df['Hippie'][X_pca_df['Hippie'] < -0.5])/len(X_pca_df)


# In[ ]:


#Narcissist
len(X_pca_df['Narcissist'][X_pca_df['Narcissist'] > 0.5])/len(X_pca_df)


# In[ ]:


len(X_pca_df['Narcissist'][X_pca_df['Narcissist'] < -0.5])/len(X_pca_df)


# <h2> Hult DNA Traits PCA 

# In[ ]:


#Removing Categorical (Demographics), SurveyID and HULT DNA from Data to generate Personality Traits only
hult_df= cust_df.copy()
hult_df = hult_df.drop(['What laptop do you currently have?', 
                            'What laptop would you buy in next assuming if all laptops cost the same?',
                            'What program are you in?', 
                            'What is your age?', 
                            'What is your ethnicity?', 
                            'Gender', 
                            'What is your nationality? ', 
                            'surveyID',
                            'Am the life of the party', 'Feel little concern for others', 
                            'Am always prepared', 'Get stressed out easily', 'Have a rich vocabulary', 
                            "Don't talk a lot", 'Am interested in people', 'Leave my belongings around', 
                            'Am relaxed most of the time', 'Have difficulty understanding abstract ideas', 
                            'Feel comfortable around people', 'Insult people', 'Pay attention to details', 
                            'Worry about things', 'Have a vivid imagination', 'Keep in the background', 
                            "Sympathize with others' feelings", 'Make a mess of things', 'Seldom feel blue', 
                            'Am not interested in abstract ideas', 'Start conversations', 
                            "Am not interested in other people's problems", 'Get chores done right away', 
                            'Am easily disturbed', 'Have excellent ideas', 'Have little to say', 
                            'Have a soft heart', 'Often forget to put things back in their proper place', 
                            'Get upset easily', 'Do not have a good imagination', 
                            'Talk to a lot of different people at parties', 
                            'Am not really interested in others', 'Like order',
       'Change my mood a lot', 'Am quick to understand things', "Don't like to draw attention to myself", 
                            'Take time out for others', 'Shirk my duties', 'Have frequent mood swings', 
                            'Use difficult words', "Don't mind being the center of attention", 
                            "Feel others' emotions", 'Follow a schedule', 'Get irritated easily', 
                            'Spend time reflecting on things', 'Am quiet around strangers', 
                            'Make people feel at ease', 'Am exacting in my work', 'Often feel blue' 
                            ],
                             axis = 1)
    


# In[ ]:


#Scaling the data

scaler = StandardScaler()
scaler.fit(hult_df)
X_scaled1 = scaler.transform(hult_df)

#Converting scaled data into a dataframe
hult_scaled = pd.DataFrame(X_scaled1)

#Reattaching column names
hult_scaled.columns = hult_df.columns

#Checking pre- and post scaling variance
print(pd.np.var(hult_df), '\n\n')
print(pd.np.var(hult_scaled))


# In[ ]:


# INSTANTIATING a PCA object with no limit to principal components
pca = PCA(n_components = None,
            random_state = 802)


# FITTING and TRANSFORMING the scaled data
hult_pca = pca.fit_transform(hult_scaled)


# comparing dimensions of each DataFrame
print("Original shape:", hult_scaled.shape)
print("PCA shape     :",  hult_pca.shape)


# In[ ]:


pd.DataFrame(hult_pca)


# In[ ]:


# component number counter
component_number = 0
sum=0
# looping over each principal component
for variance in pca.explained_variance_ratio_:
    component_number += 1
    sum=sum+variance
    print(f"PC {component_number}    : {variance.round(3)}    : {sum.round(2)}")


# In[ ]:


# printing the sum of all explained variance ratios
print(pca.explained_variance_ratio_.sum())


# <h3> Principal Component Analysis for Hult

# In[ ]:


# INSTANTIATING a PCA object with no limit to principal components
pca = PCA(n_components = None,
          random_state = 219)


# FITTING and TRANSFORMING the purchases_scaled
hult_pca = pca.fit_transform(hult_scaled)


# calling the scree_plot function
scree_plot(pca_object = pca)


# In[ ]:


#Splitting for Hult DNA traits
hult_s = hult_scaled.iloc[:, 0:10]
hult_s.head()

#Instantiating PCA object with no limit to principal components for BIG 5
hult_pca = PCA(n_components = None,
         random_state = 802)

#Fitting and transforming
hult_pca_fit = hult_pca.fit_transform(hult_s)

#Calling scree plot function
scree_plot(pca_object = hult_pca)


# In[ ]:


#PCA for BIG 5
pca_5h = PCA(n_components = 5,
           random_state = 802)

#Fitting and Transforming 
hult_pca_5 = pca_5h.fit_transform(hult_s)

#Calling the scree_plot function
scree_plot(pca_object = pca_5h)


# In[ ]:


#PCA for HULT DNA
pca_3h = PCA(n_components = 3,
           random_state = 802)

#Fitting and Transforming 
hult_pca_3 = pca_3h.fit_transform(hult_s)

#Calling the scree_plot function
scree_plot(pca_object = pca_3h)


# <h3> Hult Correlation Heatmap

# In[ ]:


# setting plot size
fig, ax = plt.subplots(figsize = (12, 12))


# developing a PC to feature heatmap
sns.heatmap(pca_3h.components_, 
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linewidths = 0.1,
            linecolor = 'black')




plt.xlabel(xlabel = "Feature")
plt.ylabel(ylabel = "Principal Component")


# displaying the plot
plt.show()


# <h3> Factor Loading - Hult

# In[ ]:


#Max PC Model for BIG 3 in hult

# transposing pca components (pc = MAX)
factor_loadings_hult = pd.DataFrame(pd.np.transpose(hult_pca.components_))


# naming rows as original features
factor_loadings_hult = factor_loadings_hult.set_index(hult_s.columns)


##################
### 5 PC Model ###
##################
# transposing pca components (pc = 5)
factor_loadings_3 = pd.DataFrame(pd.np.transpose(pca_3h.components_))


# naming rows as original features
factor_loadings_3 = factor_loadings_3.set_index(hult_s.columns)


# checking the results
print(f"""
MAX Components Factor Loadings
------------------------------
{factor_loadings_hult.round(2)}


3 Components Factor Loadings
------------------------------
{factor_loadings_3.round(2)}
""")


# In[ ]:


factor_loadings_3.columns= ['Follower', 'Creative', 'Reactive']
factor_loadings_3


# In[ ]:


# analyzing factor strengths per customer
X_pca_reduced_h = pca_3h.transform(hult_s)


# converting to a DataFrame
X_pca_df_h = pd.DataFrame(X_pca_reduced_h)


# renaming columns
X_pca_df_h.columns = factor_loadings_3.columns


# checking the results
X_pca_df_h


# Those who do not follow others and think for themselves are more likely to buy a mac. Creative users and up beat customers would buy a mac over Windows. Referring to the Hult DNA, driven and hardworking customers are looking for a product that can keep up with what they aim to achieve.

# In[ ]:


#follower
len(X_pca_df_h['Follower'][X_pca_df_h['Follower'] > 0.5])/len(X_pca_df_h)


# In[ ]:


len(X_pca_df_h['Follower'][X_pca_df_h['Follower'] < -0.5])/len(X_pca_df_h)


# In[ ]:


#Creative
len(X_pca_df_h['Creative'][X_pca_df_h['Creative'] > 0.5])/len(X_pca_df_h)


# In[ ]:


len(X_pca_df_h['Creative'][X_pca_df_h['Creative'] < -0.5])/len(X_pca_df_h)


# In[ ]:


#Reactive
len(X_pca_df_h['Reactive'][X_pca_df_h['Reactive'] > 0.5])/len(X_pca_df_h)


# In[ ]:


len(X_pca_df_h['Reactive'][X_pca_df_h['Reactive'] < -0.5])/len(X_pca_df_h)


# <h2> Clustering

# In[ ]:


# checking variance amongst clusters
np.var(X_pca_df)


# In[ ]:


# checking variance amongst clusters
np.var(X_pca_df_h)


# In[ ]:


########################################
### Personality
########################################

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with the data
scaler.fit(X_pca_df)


# TRANSFORMING our data after fit
X_scaled_pca = scaler.transform(X_pca_df)


# converting scaled data into a DataFrame
pca_scaled = pd.DataFrame(X_scaled_pca)


# reattaching column names
pca_scaled.columns = ['Introvert',      
                     'Outgoing', 
                     'Scatter Brained',   
                     'Hippie', 
                     'Narcissist'] 

# checking pre- and post-scaling variance
print(pd.np.var(X_pca_df), '\n\n')
print(pd.np.var(pca_scaled))


# In[ ]:


########################################
### HULT
########################################
# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with the data
scaler.fit(X_pca_df_h)


# TRANSFORMING our data after fit
X_scaled_pca = scaler.transform(X_pca_df_h)


# converting scaled data into a DataFrame
pca_scaled_1 = pd.DataFrame(X_scaled_pca)


# reattaching column names
pca_scaled_1.columns = ['Follower',      
                     'Creative', 
                     'Reactive'] 

# checking pre- and post-scaling variance
print(pd.np.var(X_pca_df_h), '\n\n')
print(pd.np.var(pca_scaled_1))


# <h3> Agglomerative Clustering

# Agglomerative clustering starts with each observation in its own cluster. We are using the WARD method  for calculating the distance which  will  minimizes the variance amongst all clusters and leads to clusters that are relatively equal in size. Our number of clusters cuts in half as certain colors seem to create stronger niches than others.

# In[ ]:


# grouping data for personality
standard_mergings_pers = linkage(y = pca_scaled,
                                 method = 'ward',
                                 optimal_ordering = True)


# setting plot size
fig, ax = plt.subplots(figsize=(12, 12))

# developing a dendrogram
dendrogram(Z = standard_mergings_pers,
           leaf_rotation = 90,
           leaf_font_size = 6)


# saving and displaying the plot
#plt.savefig('./analysis_images/standard_hierarchical_clust_ward.png')
plt.show()


# Personality reflects on this as our clusters are based off of the variance within these traits, showing orange and green (the first two) as potential strong niches. 

# In[ ]:


# grouping data for personality
standard_mergings_hult = linkage(y = pca_scaled_1,
                                 method = 'ward',
                                 optimal_ordering = True)


# setting plot size
fig, ax = plt.subplots(figsize=(12, 12))

# developing a dendrogram
dendrogram(Z = standard_mergings_hult,
           leaf_rotation = 90,
           leaf_font_size = 6)


# saving and displaying the plot
#plt.savefig('./analysis_images/standard_hierarchical_clust_ward.png')
plt.show()


# The first cluster creates their own specific niche seen in hult DNA. Three clusters seem to be the strongest for this dataframe.

# In[ ]:


# calling the inertia_plot() function for personality
interia_plot(data = pca_scaled)


# In[ ]:


# calling the inertia_plot() function for huly traits
interia_plot(data = pca_scaled_1)


# In[ ]:


# INSTANTIATING a k-Means object with clusters for personality
customers_k_pca = KMeans(n_clusters   = 6,
                         random_state = 219)


# fitting the object to the data
customers_k_pca.fit(pca_scaled)


# converting the clusters to a DataFrame
customers_kmeans_pca = pd.DataFrame({'Cluster_p': customers_k_pca.labels_})


# checking the results
print(customers_kmeans_pca.iloc[: , 0].value_counts())


# In[ ]:


# INSTANTIATING a k-Means object with clusters for hult trains
customers_k_pca_h = KMeans(n_clusters   = 5,
                         random_state = 219)


# fitting the object to the data
customers_k_pca_h.fit(pca_scaled_1)


# converting the clusters to a DataFrame
customers_kmeans_pca_h = pd.DataFrame({'Cluster_h': customers_k_pca_h.labels_})


# checking the results
print(customers_kmeans_pca_h.iloc[: , 0].value_counts())


# <h3> Storing Cluster Centers for Personality and Hult

# In[ ]:


# storing cluster centers- personality
centroids_pca = customers_k_pca.cluster_centers_


# converting cluster centers into a DataFrame
centroids_pca_df = pd.DataFrame(centroids_pca)


# renaming principal components
centroids_pca_df.columns = ['Introvert',      
                     'Outgoing', 
                     'Scatter Brained',   
                     'Hippie', 
                     'Narcissist'] 


# checking results (clusters = rows, pc = columns)
centroids_pca_df.round(2)


# In[ ]:


# storing cluster centers- hult
centroids_pca_h = customers_k_pca_h.cluster_centers_


# converting cluster centers into a DataFrame
centroids_pca_df_h = pd.DataFrame(centroids_pca_h)


# renaming principal components
centroids_pca_df_h.columns = ['Follower',      
                     'Creative', 
                     'Reactive'] 


# checking results (clusters = rows, pc = columns)
centroids_pca_df_h.round(2)


# In[ ]:


X_pca_df_h


# <h3> Concatenate Both Clusters 

# In[ ]:


#Concatenate demographics, HULT and personality components into one DataFrame.
# concatenating cluster memberships with principal components
clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_df, ],
                         axis = 1)


# checking results
#clst_pca_df

# concatenating cluster memberships with principal components
#clst_pca_df_2 = pd.concat([customers_kmeans_pca_h,
                          #X_pca_df_h ],
                         #axis = 1)


# checking results
#clst_pca_df_2

# concatenating demographic information with pca-clusters
final_pca_clust_df = pd.concat([categ_df.iloc[:,:],clst_pca_df
                                ],
                                axis = 1)

# renaming columns
final_pca_clust_df.columns = ['surveyID', 
                            'What laptop do you currently have?', 
                            'What laptop would you buy in next assuming if all laptops cost the same?',
                            'What program are you in?', 
                            'What is your age?', 
                            'Gender',
                            'What is your nationality? ',
                            'What is your ethnicity?', 
                            'Cluster_p',
                            'Introvert',      
                             'Outgoing', 
                             'Scatter Brained',   
                             'Hippie', 
                             'Narcissist']


final_pca_clust_df


# In[ ]:


# checking results
#clst_pca_df

# concatenating cluster memberships with principal components
clst_pca_df_2 = pd.concat([customers_kmeans_pca_h,
                          X_pca_df_h ],
                         axis = 1)

final_pca_clust_df2 = pd.concat([final_pca_clust_df,clst_pca_df_2
                                ],

                            axis = 1)

# renaming columns
final_pca_clust_df2


# In[ ]:


#final pc clust
final_pca_clust_df2= final_pca_clust_df2.dropna()
final_pca_clust_df2


# In[ ]:


#Cluster Names 
# renaming cluster
cluster_names = {0 : 'Cluster 1',
                 1 : 'Cluster 2',
                 2 : 'Cluster 3',
                 3 : 'Cluster 4',
                 4 : 'Cluster 5',
                 5 : 'Cluster 6'}


final_pca_clust_df2['Cluster_p'].replace(cluster_names, inplace = True)

# renaming cluster
cluster_names = {0 : 'Cluster 1',
                 1 : 'Cluster 2',
                 2 : 'Cluster 3',
                 3 : 'Cluster 4',
                 4 : 'Cluster 5'}


final_pca_clust_df2['Cluster_h'].replace(cluster_names, inplace = True)

# adding a productivity step
data_df = final_pca_clust_df2


# checking results
data_df


# <h2> Demographic Analysis

# <h3> What laptop do you currently have?

# In[ ]:


########################
# What laptop do you currently have?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# **Follower's analysis**
# Clusters 2 and 4 have a tendency to not be followers. There is no distinction between Macbook or PC users. 
# 
# **Creative's analysis**
# Cluster 1 and 3 have a negative correlation to being creative. There is no distinction between Macbook or PC users. 
# 
# **Reactive's analysis**
# Most clusters are indifferent in regards to their reactiveness without any distinction between Macbook or PC users. Besides cluster 1 who is less reactive.  

# In[ ]:


########################
# What laptop do you currently have?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop do you currently have?',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# **Introvert's analysis**
# Cluster 6 is more likely to be a Macbook user, with a less introverted personality.
# 
# **Outgoing's analysis**
# Clusters 2, 3 and 6 are the most aligned with the outgoing personality but no distinction in if they're Macbook or PC users. Windows users in cluster 4 differ from Mac users in that Windows users seem to be more outgoing.  
# 
# **Scatter Brained analysis**
# Personality don't seem to change depending on if they are a Macbook or a Windows user. 
# 
# **Hippie**
# Clusters 4 and 5 tend to vary more depending if they're Macbook or Windows users regarding in how well their personality aligns with them. For example for a Macbook user in Cluster 4 the tendency of having "Hippie" characteristics is negative correlated in comparison to a PC user which are positively correlated.
# 
# **Narcissists**
# Most clusters seem to be aligned with Narcissists characteristics except for cluster 3, which without any distinction in being a Macbook user or PC user, is not aligned with Narcissists characteristics.  

# <h3> What laptop would you buy in next assuming if all laptops cost the same?

# In[ ]:


########################
# What laptop would you buy in next assuming if all laptops cost the same?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# **Follower's analysis**
# Clusters 2 and 4 seem to not to be aligned with "Follower's" characteristics in comparison to the rest that are either aligned or positive correlated.
# 
# **Creative's analysis**
# Clusters 1 and 3 don't seem to share all the "Creative's" characteristics (negatively correlated), while only cluster 5 would change its characteristics according to if it is or is not a Window's or Mac user.  In clusters 4 & 5 we can observe that people are slightly more creative while using Macbooks. 
# 
# **Reactive's analysis**
# In general reactive personalities tend to be more aligned without any distinction in if they're Window's or Macbook users. The only cluster that does not follow this approach is Cluster 1, which actually shows a negative tendency either in Macbook or Windows users. 

# In[ ]:


########################
# What laptop would you buy in next assuming if all laptops cost the same?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What laptop would you buy in next assuming if all laptops cost the same?',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# **Introvert's analysis**
# These clusters seem to vary according to their characteristics whether they are Macbook or Windows users. In general this group would need extra effort to be identified according to their characteristics.   
# 
# **Outgoing's analysis** 
# Cluster 4 seems to have different characteristics based on if its a Windows or PC user. 
# Cluster does not share "Outgoing" characteristics in either of the two segmentations (Windows or PC's user).  
# 
# **Scatter Brained analysis**
# Scatter Brained profiles are largely Macbook or Window's users.  
# 
# **Hippie**
# Less of "Hippies" in Cluster 2 and 4 seem to prefer Macbook. 
# 
# **Narcissists** 
# Most of the clusters are close to the center or have a positive tendency towards the "Narcissist" category, except for the Cluster 3, that does not align with those characteristics. 

# <h3> What program are you in?

# In[ ]:


########################
# What program are you in?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# In[ ]:


########################
# What program are you in?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What program are you in?',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# **What program are you in?**
# 
# They don't let us fairly analyze programs, as they're not evenly categorized. We can observe that as there are three dual degree categories, but only one single year category. If there is variation in categorized dual degree options then there must also be variation in single year options so that you can't properly see clear patterns. It may also be concluded that this survey was done primarily on dual degree students. Business analytics kids do seem to be a bit narcissistic though. ;)

# <h3> What is your age?

# In[ ]:


########################
# What is your age?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# The age gap that we're looking at ,we assume, is college and Master's students.
# 
# The only pattern we can observe is for followers and only within clusters 1 & 5, which have a high inclination towards being followers. The rest of the categories don't seem to have a clear clustered pattern. Therefore, we can't say that age is associated with the Hult DNA.

# In[ ]:


########################
#  What is your age?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your age?',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# From the boxplots displayed, we can say that age is not a determinant factor of someone's personality. For example, we could have a Narcisist personality at the age of 20 or even at the age of 34.

# <h3> What is your ethnicity?

# In[ ]:


########################
# What is your ethnicity?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# In[ ]:


########################
# What is your ethnicity?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your ethnicity?',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# Patterns vary a lot by looking at the ethnicity. It is challenging to get a real perspective of patterns amongst all of the personality traits and ethnicity. There's another important factor that we should take into consideration while analyzing data by this trait, as we don't have equal proportional data in terms of ethnicity and as we are not aware of where this survey took place, therefore ethnicity would vary.

# <h3> Gender

# In[ ]:


########################
# Gender
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# For gender and the Hult DNA traits, we can conclude that some of the characteristics such as the "Creative" and "Reactive" characteristics are not affected by gender, while the characteristic of "Followers" does vary by gender. For the Hult DNA, it may not be in the best interest to perform an analysis segments by gender. Moreover, making distinctions by gender could lead to misleading information as it could be a sensitive topic and information for the person answering the survey and as people could prefer not to answer this question and therefore this wouldn't be representative of the full sample.

# In[ ]:


########################
#Gender
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Outgoing',
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'Gender',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# The type of personality does seem to be affected by gender as we can observe variation in most of the medians of the boxplots, however, companies must analyze the efforts that further analysis may need.

# <h3> What is your nationality?

# In[ ]:


########################
#What is your nationality?
########################
# Follower
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Follower',
            hue  = 'Cluster_h',
            data = data_df)

# Creative
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Creative',
            hue  = 'Cluster_h',
            data = data_df)

# Reactive
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Reactive',
            hue  = 'Cluster_h',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# Nationality is not a proper metric to be evaluated in conjunction to the Hult DNA due to its broaden approach.

# In[ ]:


########################
# What laptop do you currently have?
########################
# Introvert
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Introvert',
            hue  = 'Cluster_p',
            data = data_df)

# Outgoing
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Outgoing', 
            hue  = 'Cluster_p',
            data = data_df)

# Scatter Brained
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Scatter Brained',
            hue  = 'Cluster_p',
            data = data_df)

# Hippie
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Hippie',
            hue  = 'Cluster_p',
            data = data_df)

# Narcissist
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x    = 'What is your nationality? ',
            y    = 'Narcissist',
            hue  = 'Cluster_p',
            data = data_df)

# formatting and displaying the plot
plt.tight_layout()
plt.show()


# Nationality is such a specific category that it is challenging to use it as an indicator. In the future it will be useful to create subgroups such as region in order to have a better understanding of customer type. 
# There is a distinct audience seen in personality and Hult DNA as we look at nationalities especially on the left side of the graph, but specific patterns would come from subgroups.  
# 

# <h2> Conclusions

# Through our findings, we inferred  that the group of people being surveyed were potential college and masters students. Some bias may have occurred from the categorical variables as age group, program, and nationality either did not have enough variation, or there was too much to fairly analyze it (i.e. not many people above 40 were in this survey).
# 
# Externally, a large amount of college students usually have Macbooks due to their user friendly characteristics and overall popularity seen in Apple products. This was proven in the dataset as in if there was any variation at all, it usually pointed towards Macbook. For example, we found that creative people tend to prefer them.
# 
# Through our findings, we inferred that the group of people being surveyed were potential college and masters students. Some bias may have occurred from the categorical variables as age group, program, and nationality either did not have enough variation, or there was too much to fairly analyze it (i.e. not many people above 40 were in this survey).
# 
# Externally, a large amount of college students usually have Macbooks due to their user friendly characteristics and overall popularity seen in Apple products. This was proven in the dataset as in if there was any variation at all, it usually pointed towards Macbook. For example, we found that creative and narcissitic people tend to prefer them.  Also in our findings  from our personality data, creative people seen in the Hult DNA  were most responsive. 
# 
# As most of the data falls in favor of Apple, there is still room for more improvement. An example of this was seen when if someone already owned a macbook and was part of the survey, there is a high likelihood they would buy another. Apple may want to survey people who don’t own a macbook. In assuming they conducted this survey to look at students, there should be more categories in programs to generate balance and fairness.
