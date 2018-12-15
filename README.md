
<center>
    <b>
        <font size="+3">
            Food Pairing and Data Science
        </font>
    </b>
    <br>
    <br>
    Vincent Choo
</center>

# Introduction

Food pairing is...

# Acquiring Data

First, we'll need a database of flavor compounds in each kind of food ingredient.

Several databases exist, such as [FoodDB](http://foodb.ca/), [FlavorNet](http://www.flavornet.org/), and [FlavorDB](https://www.ncbi.nlm.nih.gov/pubmed/29059383), but not all associate foods with the compounds they contain. The one at FlavorDB does, so we scrape our data from the FlavorDB [website](https://cosylab.iiitd.edu.in/flavordb/) to generate [Pandas DataFrames](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html).

The data at FlavorDB is scattered across JSON files on the site, where each JSON file corresponds to a particular food and the flavor molecules in it. Fortunately, the files each have a numerical id, so we can grab the JSON files by iterating over URLs of the right form.


```python
# import the relevant Python packages
import urllib.request
import json

import numpy as np
import pandas as pd
import math
```


```python
# JSON files are at addresses of this form
def flavordb_entity_url(x):
    return "https://cosylab.iiitd.edu.in/flavordb/entities_json?id="+str(x)

# translates the JSON file at the specified web address into a dictionary
def get_flavordb_entity(x):
    with urllib.request.urlopen(flavordb_entity_url(x)) as url:
        return json.loads(url.read().decode())
    return None
```

Then, we convert the JSON files into ``DataFrames``.


```python
# the names of the columns in the raw JSON objects
def flavordb_entity_cols():
    return [
        'entity_id', 'entity_alias_readable', 'entity_alias_synonyms',
        'natural_source_name', 'category_readable', 'molecules'
    ]


# define the names of the columns in the dataframes we want to generate
def flavordb_df_cols():
    return [
        'entity id', 'alias', 'synonyms',
        'scientific name', 'category', 'molecules'
    ]


def molecules_df_cols():
    return ['pubchem id', 'common name', 'flavor profile']    
```


```python
def clean_flavordb_dataframes(flavor_df, molecules_df):
    strtype = type('')
    settype = type(set())
    
    for k in ['alias', 'scientific name', 'category']:
        flavor_df[k] = [
            elem.lower() if isinstance(elem, strtype) else ''
            for elem in flavor_df[k]
        ]
    
    flavor_df['synonyms'] = [
        elem if isinstance(elem, settype) else (
            set(elem.lower().split(', ') if isinstance(elem, strtype) else [''])
        )
        for elem in flavor_df['synonyms']
    ]
    
    molecules_df['flavor profile'] = [
        set([x.lower() for x in elem])
        for elem in molecules_df['flavor profile']
    ]
    
    return flavor_df, molecules_df
```


```python
# generate dataframes from some of the JSON objects
def get_flavordb_dataframes(start, end):
    # make intermediate values to make dataframes from
    flavordb_data = []
    molecules_dict = {}
    missing = [] # numbers of the missing JSON files during iteration
    
    flavordb_cols = flavordb_entity_cols()
    
    for i in range(start, end):
        # get the ith food entity, as a JSON dict
        try:
            fdbe = get_flavordb_entity(i + 1)

            # get only the relevant fields (columns) of the dict
            flavordb_series = [fdbe[k] for k in flavordb_cols[:-1]]
            flavordb_series.append(
                set([m['pubchem_id'] for m in fdbe['molecules']])
            )
            flavordb_data.append(flavordb_series)

            # update the molecules database with the data in 'molecules' field
            for m in fdbe['molecules']:
                if m['pubchem_id'] not in molecules_dict:
                    molecules_dict[m['pubchem_id']] = [
                        m['common_name'],
                        set(m['flavor_profile'].split('@'))
                    ]
        except urllib.error.HTTPError as e:
            if e.code == 404: # if the JSON file is missing
                missing.append(i)
            else:
                raise RuntimeError(
                    'Error while fetching JSON object from ' + flavordb_entity_url(x)
                ) from e
            

    # generate the dataframes
    flavordb_df = pd.DataFrame(
        flavordb_data,
        columns=flavordb_df_cols()
    )
    molecules_df = pd.DataFrame(
        [
            [k, v[0], v[1]]
             for k, v in molecules_dict.items()
        ],
        columns=molecules_df_cols()
    )
    
    # clean up the dataframe columns
    flavordb_df, molecules_df = clean_flavordb_dataframes(flavordb_df, molecules_df)
    
    return [flavordb_df, molecules_df, missing]
```

It takes a while to download all of these JSON files, so make sure to save your download progress!


```python
# updates & saves the download progress of your dataframes
def update_flavordb_dataframes(df0, df1, ranges):
    df0_old = df0
    df1_old = df1
    missing_old = []

    # time how long it took to download the files
    import time
    start = time.time()

    # save the download progress in increments of 50 JSON files
    try:
        # as of today, it looks like FlavorDB has about 1000 distinct entities
        for a, b in ranges:
            df0_new, df1_new, missing_new = get_flavordb_dataframes(a, b)
            
            df0_old = df0_old.append(df0_new, ignore_index=True)
            df1_old = df1_old.append(df1_new, ignore_index=True)
            missing_old.extend(missing_new)
        
        return df0_old, df1_old, missing_old
    except:
        raise # always throw the error so you know what happened
    finally:
        # even if you throw an error, you'll have saved them as csv files
        df0_old.to_csv('flavordb.csv')
        df1_old.to_csv('molecules.csv')

        end = time.time()
        mins = (end - start) / 60.0
        print('Downloading took: '+ str(mins) + ' minutes')
```


```python
# take new dataframes
df0 = pd.DataFrame(columns=flavordb_df_cols())
df1 = pd.DataFrame(columns=molecules_df_cols())
# fill them with JSON files up to id = 1000
ranges = [(50 * i, 50 * (i + 1)) for i in range(20)]
# update & save the dataframes as csv files
update_flavordb_dataframes(df0, df1, ranges)
```

While downloading the JSON files, you'll notice that some of them are missing due to ``HTTPError``s. The first time you download, you might, say, notice that 43 entries are missing.


```python
# get the missing entries
def missing_entity_ids(flavor_df):
    out = []
    entity_id_set = set(flavor_df['entity id'])
    for i in range(1, 1 + max(entity_id_set)):
        if i not in entity_id_set:
            out.append(i)
    return out


# loads the dataframes from csv files
def load_db():
    settype = type(set())
    
    df0 = pd.read_csv('flavordb.csv')[flavordb_df_cols()]
    df0['synonyms'] = [eval(x) if isinstance(x, settype) else x for x in df0['synonyms']]
    df0['molecules'] = [eval(x) for x in df0['molecules']]
    
    df1 = pd.read_csv('molecules.csv')[molecules_df_cols()]
    df1['flavor profile'] = [eval(x) for x in df1['flavor profile']]
    
    df0, df1 = clean_flavordb_dataframes(df0, df1)
    
    return df0, df1, missing_entity_ids(df0)
```


```python
# missing_ids = the missing ids that are less than the max one found
flavor_df, molecules_df, missing_ids = load_db()
flavor_df.to_csv('flavordb.csv')
molecules_df.to_csv('molecules.csv')
flavor_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity id</th>
      <th>alias</th>
      <th>synonyms</th>
      <th>scientific name</th>
      <th>category</th>
      <th>molecules</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>bakery products</td>
      <td>{bakery products}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{27457, 7976, 31252, 26808, 22201, 26331}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>bread</td>
      <td>{bread}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{1031, 1032, 644104, 527, 8723, 31260, 15394, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>rye bread</td>
      <td>{rye bread}</td>
      <td>rye</td>
      <td>bakery</td>
      <td>{644104, 7824, 643731, 8468, 1049, 5372954, 80...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>wheaten bread</td>
      <td>{soda scones, soda farls}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{6915, 5365891, 12170, 8082, 31251, 7958, 1049...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>white bread</td>
      <td>{white bread}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{7361, 994, 7362, 10883, 11173, 5365891, 11559...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>wholewheat bread</td>
      <td>{wholewheat bread}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{107905, 8194, 10883, 13187, 5283329, 5283335,...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>wort</td>
      <td>{wort}</td>
      <td>barley</td>
      <td>beverage</td>
      <td>{13187, 9862, 135, 18827, 7824, 61712, 19602, ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>arrack</td>
      <td>{arak}</td>
      <td>grape</td>
      <td>beverage alcoholic</td>
      <td>{1031, 240, 31249, 6584, 7997}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>beer</td>
      <td>{beer}</td>
      <td>poacceae</td>
      <td>beverage alcoholic</td>
      <td>{229888, 62465, 8194, 8193, 1031, 644104, 5283...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>bantu beer</td>
      <td>{pombe, millet beer, malwa, kaffir beer, opaqu...</td>
      <td>eragrostideae</td>
      <td>beverage alcoholic</td>
      <td>{6560, 8038, 7654, 7147, 1068, 14286, 527, 240...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>brandy</td>
      <td>{ kanyak, armagnac, konyak , pisco, cognac, st...</td>
      <td>vitis vinifera</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 8468, 1049, 531804...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>anise brandy</td>
      <td>{brandy}</td>
      <td>anise</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 1031, 8468, 1049, ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>apple brandy</td>
      <td>{anise brandy}</td>
      <td>apple</td>
      <td>beverage alcoholic</td>
      <td>{62465, 10885, 9862, 263, 1031, 8073, 5364231,...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>armagnac brandy</td>
      <td>{armanac brandy}</td>
      <td>vitis vinifera</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 1031, 8468, 1049, ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>blackberry brandy</td>
      <td>{brackberry brandy}</td>
      <td>blackberry</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 8468, 1049, 531804...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>cherry brandy</td>
      <td>{cherry brandy}</td>
      <td>prunus</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 31249, 8468, 1049,...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>cognac brandy</td>
      <td>{cognac brandy}</td>
      <td>vitis</td>
      <td>beverage alcoholic</td>
      <td>{62465, 12293, 1031, 5283335, 5364231, 31242, ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>papaya brandy</td>
      <td>{papaya brandy}</td>
      <td>papaya</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 8468, 1049, 531804...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>pear brandy</td>
      <td>{poire williams, rakia, pear brandy, tuica, pa...</td>
      <td>pear</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 1031, 5281162, 528...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>plum brandy</td>
      <td>{slivovitz, ljivovica, schlivowitz, plum brand...</td>
      <td>damson</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 1031, 527, 31249, ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>raspberry brandy</td>
      <td>{raspberry brandy}</td>
      <td>rubus idaeus</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 8468, 1049, 531804...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>weinbrand brandy</td>
      <td>{weinbrand brandy}</td>
      <td>vitis vinifera</td>
      <td>beverage alcoholic</td>
      <td>{62465, 5364231, 263, 8073, 8468, 1049, 531804...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>gin</td>
      <td>{gin}</td>
      <td>juniperus communis</td>
      <td>beverage alcoholic</td>
      <td>{6560, 7460, 8294, 8103, 644104, 1031, 1130, 6...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>rum</td>
      <td>{rum}</td>
      <td>saccharum</td>
      <td>beverage alcoholic</td>
      <td>{229888, 62465, 8194, 31234, 229377, 12293, 10...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>whisky</td>
      <td>{whisky}</td>
      <td>maize</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 644104, 1032, 31242, 527,...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>bourbon whisky</td>
      <td>{bourbon whisky}</td>
      <td>corn</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 1031, 1032, 31242, 527, 3...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>canadian whisky</td>
      <td>{canadian whisky}</td>
      <td>corn</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 1032, 31242, 527, 31249, ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>finnish whisky</td>
      <td>{finnish whisky}</td>
      <td>maize</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 1032, 31242, 527, 31249, ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>japanese whisky</td>
      <td>{japanese whisky}</td>
      <td>maize</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 1032, 31242, 527, 31249, ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>malt whisky</td>
      <td>{malt whisky}</td>
      <td>maize</td>
      <td>beverage alcoholic</td>
      <td>{62465, 8193, 12293, 1031, 1032, 31242, 85519,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>905</th>
      <td>942</td>
      <td>horchata</td>
      <td>{orxata }</td>
      <td></td>
      <td>beverage</td>
      <td>{644104, 1130, 8094}</td>
    </tr>
    <tr>
      <th>906</th>
      <td>943</td>
      <td>soft drink</td>
      <td>{}</td>
      <td></td>
      <td>beverage</td>
      <td>{644104, 1130, 247, 6202, 8094}</td>
    </tr>
    <tr>
      <th>907</th>
      <td>944</td>
      <td>milkshake</td>
      <td>{}</td>
      <td></td>
      <td>beverage</td>
      <td>{644104, 1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>908</th>
      <td>945</td>
      <td>chocolate mousse</td>
      <td>{}</td>
      <td>theobroma</td>
      <td>bakery</td>
      <td>{644104, 1130, 8094}</td>
    </tr>
    <tr>
      <th>909</th>
      <td>947</td>
      <td>pupusa</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>910</th>
      <td>948</td>
      <td>empanada</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{1130, 8094}</td>
    </tr>
    <tr>
      <th>911</th>
      <td>949</td>
      <td>arepa</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>912</th>
      <td>950</td>
      <td>ascidians</td>
      <td>{sea squirts, ascidians}</td>
      <td>tunicate</td>
      <td>seafood</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>913</th>
      <td>951</td>
      <td>gefilte fish</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>914</th>
      <td>952</td>
      <td>yellow pond lily</td>
      <td>{brandy bottle}</td>
      <td>nuphar</td>
      <td>plant</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 439341...</td>
    </tr>
    <tr>
      <th>915</th>
      <td>953</td>
      <td>fish burger</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>916</th>
      <td>954</td>
      <td>other dish</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130, 247, 6202, 8094}</td>
    </tr>
    <tr>
      <th>917</th>
      <td>955</td>
      <td>pot pie</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>918</th>
      <td>956</td>
      <td>stuffing</td>
      <td>{}</td>
      <td></td>
      <td>additive</td>
      <td>{644104, 1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>919</th>
      <td>958</td>
      <td>fudge</td>
      <td>{}</td>
      <td>cattle</td>
      <td>bakery</td>
      <td>{644104, 1130, 8094}</td>
    </tr>
    <tr>
      <th>920</th>
      <td>959</td>
      <td>candy bar</td>
      <td>{}</td>
      <td>theobroma</td>
      <td>bakery</td>
      <td>{644104, 1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>921</th>
      <td>960</td>
      <td>condensed milk</td>
      <td>{}</td>
      <td>cattle</td>
      <td>dairy</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>922</th>
      <td>961</td>
      <td>margarine</td>
      <td>{}</td>
      <td>vegetable oil</td>
      <td>additive</td>
      <td>{644104, 1130, 6202, 8094}</td>
    </tr>
    <tr>
      <th>923</th>
      <td>962</td>
      <td>margarine like spread</td>
      <td>{}</td>
      <td></td>
      <td>additive</td>
      <td>{644104, 1130, 247, 6202, 8094}</td>
    </tr>
    <tr>
      <th>924</th>
      <td>963</td>
      <td>hummus</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130, 8094}</td>
    </tr>
    <tr>
      <th>925</th>
      <td>964</td>
      <td>potato puffs</td>
      <td>{}</td>
      <td></td>
      <td>dish</td>
      <td>{644104, 1130, 8094, 247}</td>
    </tr>
    <tr>
      <th>926</th>
      <td>965</td>
      <td>potato gratin</td>
      <td>{}</td>
      <td>potato</td>
      <td>dish</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>927</th>
      <td>967</td>
      <td>chinese bayberry</td>
      <td>{chinese strawberry, mountain peach, yamamomo,...</td>
      <td>myrica</td>
      <td>berry</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 439341...</td>
    </tr>
    <tr>
      <th>928</th>
      <td>968</td>
      <td>green zucchini</td>
      <td>{courgette}</td>
      <td>cucurbita_pepo</td>
      <td>vegetable</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 65064,...</td>
    </tr>
    <tr>
      <th>929</th>
      <td>969</td>
      <td>yellow zucchini</td>
      <td>{yellow zucchini}</td>
      <td>cucurbita_pepo</td>
      <td>vegetable</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 65064,...</td>
    </tr>
    <tr>
      <th>930</th>
      <td>970</td>
      <td>saskatoon berry</td>
      <td>{alder-leaf shadbush, chuckley pear, western j...</td>
      <td>amelanchier</td>
      <td>berry</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 439341...</td>
    </tr>
    <tr>
      <th>931</th>
      <td>971</td>
      <td>nanking cherry</td>
      <td>{shanghai cherry, mountain cherry, chinese bus...</td>
      <td>prunus cerasus</td>
      <td>berry</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 439341...</td>
    </tr>
    <tr>
      <th>932</th>
      <td>972</td>
      <td>japanese pumpkin</td>
      <td>{japanese pumpkin, kabocha}</td>
      <td>winter squash</td>
      <td>fruit</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 439341...</td>
    </tr>
    <tr>
      <th>933</th>
      <td>977</td>
      <td>guinea hen</td>
      <td>{guinea fowl, original fowl, pet speckled hen}</td>
      <td>galliformes</td>
      <td>meat</td>
      <td>{644104, 1130}</td>
    </tr>
    <tr>
      <th>934</th>
      <td>978</td>
      <td>cucurbita</td>
      <td>{cucurbita}</td>
      <td>cucurbita</td>
      <td>gourd</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 65064,...</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 6 columns</p>
</div>




```python
molecules_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pubchem id</th>
      <th>common name</th>
      <th>flavor profile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22201</td>
      <td>2,3-Dimethylpyrazine</td>
      <td>{peanut, peanut butter, butter, cocoa, leather...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31252</td>
      <td>2,5-Dimethylpyrazine</td>
      <td>{medicine, roasted nuts, roast beef, woody, co...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26331</td>
      <td>2-Ethylpyrazine</td>
      <td>{peanut, peanut butter, butter, bitter, woody,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27457</td>
      <td>2-Ethyl-3-Methylpyrazine</td>
      <td>{peanut, earthy, roast, hazelnut, corn, potato...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7976</td>
      <td>2-Methylpyrazine</td>
      <td>{peanut, chocolate, green, cocoa, popcorn, roa...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26808</td>
      <td>2,3,5-Trimethylpyrazine</td>
      <td>{peanut, earthy, roast, hazelnut, cocoa, potat...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>323</td>
      <td>coumarin</td>
      <td>{sweet, new mown hay, bitter, green, tonka}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7150</td>
      <td>Methyl Benzoate</td>
      <td>{sweet, prune, floral, herb, lettuce, cananga,...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11509</td>
      <td>3-Hexanone</td>
      <td>{ether, sweet, grape, waxy, fruity, rum}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>637566</td>
      <td>Geraniol</td>
      <td>{rose, sweet, floral, geranium, citrus, waxy, ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>439341</td>
      <td>alpha-Maltose</td>
      <td>{sweet, odorless}</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33931</td>
      <td>2-Ethyl-4-hydroxy-5-methyl-3(2H)-furanone</td>
      <td>{sweet, butterscotch, caramel, candy}</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9261</td>
      <td>Pyrazine</td>
      <td>{pungent, sweet corn, hazelnut, barley, roasted}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6072</td>
      <td>Phlorizin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12587</td>
      <td>4-Methylpentanoic Acid</td>
      <td>{pungent, cheese}</td>
    </tr>
    <tr>
      <th>15</th>
      <td>878</td>
      <td>methanethiol</td>
      <td>{gasoline, garlic, decomposing cabbage, sulfur}</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5281708</td>
      <td>daidzein</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>17</th>
      <td>240</td>
      <td>benzaldehyde</td>
      <td>{sweet, strong, cherry, bitter, burnt sugar, s...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>244</td>
      <td>benzyl alcohol</td>
      <td>{rose, grapefruit, sweet, floral, berry, cherr...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>441484</td>
      <td>alpha-L-Sorbopyranose</td>
      <td>{sweet, bitter}</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1889</td>
      <td>DL-Liquiritigenin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4788</td>
      <td>phloretin</td>
      <td>{odorless, bitter}</td>
    </tr>
    <tr>
      <th>22</th>
      <td>6184</td>
      <td>Hexanal</td>
      <td>{fatty, aldehydic, green, sweaty, fat, grass, ...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>11128</td>
      <td>Linamarin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8723</td>
      <td>2-Methyl-1-Butanol</td>
      <td>{onion, malt, wine, roasted, fruity}</td>
    </tr>
    <tr>
      <th>25</th>
      <td>12756</td>
      <td>gamma-Caprolactone</td>
      <td>{sweet, herbal, coconut, tobacco, coumarin}</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8094</td>
      <td>Heptanoic Acid</td>
      <td>{rancid, cheese, cheesy, sour, sweat}</td>
    </tr>
    <tr>
      <th>27</th>
      <td>338</td>
      <td>salicylic acid</td>
      <td>{phenolic, nutty, faint}</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11005</td>
      <td>Tetradecanoic acid</td>
      <td>{waxy, fatty, coconut, soapy}</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1183</td>
      <td>vanillin</td>
      <td>{sweet, creamy, vanilla, chocolate}</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7593</th>
      <td>6050</td>
      <td>Tributyrin</td>
      <td>{fatty, cheese, bitter, creamy, waxy}</td>
    </tr>
    <tr>
      <th>7594</th>
      <td>8130</td>
      <td>Heptanal</td>
      <td>{wine-lee, fatty, rancid, herbal, aldehydic, o...</td>
    </tr>
    <tr>
      <th>7595</th>
      <td>13144</td>
      <td>calcium lactate</td>
      <td>{odorless}</td>
    </tr>
    <tr>
      <th>7596</th>
      <td>180</td>
      <td>acetone</td>
      <td>{apple, pear, ethereal, solvent}</td>
    </tr>
    <tr>
      <th>7597</th>
      <td>6569</td>
      <td>2-Butanone</td>
      <td>{ether, ethereal, acetone, camphor, fruity}</td>
    </tr>
    <tr>
      <th>7598</th>
      <td>650</td>
      <td>2,3-butanedione</td>
      <td>{pungent, sweet, oily, strong, butter, creamy,...</td>
    </tr>
    <tr>
      <th>7599</th>
      <td>61503</td>
      <td>D-Lactic acid</td>
      <td>{odorless, acidic}</td>
    </tr>
    <tr>
      <th>7600</th>
      <td>638278</td>
      <td>isoliquiritigenin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7601</th>
      <td>126</td>
      <td>4-hydroxybenzaldehyde</td>
      <td>{sweet, woody, balsam, almond, nutty}</td>
    </tr>
    <tr>
      <th>7602</th>
      <td>444539</td>
      <td>Cinnamic Acid</td>
      <td>{sweet, storax, honey, balsam}</td>
    </tr>
    <tr>
      <th>7603</th>
      <td>5280598</td>
      <td>Farnesal</td>
      <td>{floral, minty}</td>
    </tr>
    <tr>
      <th>7604</th>
      <td>5280445</td>
      <td>luteolin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7605</th>
      <td>8768</td>
      <td>3,4-Dihydroxybenzaldehyde</td>
      <td>{dry, bitter, almond, medical}</td>
    </tr>
    <tr>
      <th>7606</th>
      <td>637542</td>
      <td>p-coumaric acid</td>
      <td>{balsamic, balsam}</td>
    </tr>
    <tr>
      <th>7607</th>
      <td>11552</td>
      <td>3-Methylbutanal</td>
      <td>{fatty, ethereal, chocolate, malt, aldehydic, ...</td>
    </tr>
    <tr>
      <th>7608</th>
      <td>445070</td>
      <td>farnesol</td>
      <td>{sweet, grapefruit, floral, anise, muguet, mil...</td>
    </tr>
    <tr>
      <th>7609</th>
      <td>6202</td>
      <td>Thiamine Hydrochloride</td>
      <td>{bitter, sour, mild}</td>
    </tr>
    <tr>
      <th>7610</th>
      <td>8468</td>
      <td>Vanillic acid</td>
      <td>{sweet, dairy, bean, powdery, milky, creamy, v...</td>
    </tr>
    <tr>
      <th>7611</th>
      <td>72277</td>
      <td>(-)-Epigallocatechin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7612</th>
      <td>602</td>
      <td>Dl-Alanine</td>
      <td>{odorless}</td>
    </tr>
    <tr>
      <th>7613</th>
      <td>876</td>
      <td>Dl-Methionine</td>
      <td>{sulfurous, mild, acidic}</td>
    </tr>
    <tr>
      <th>7614</th>
      <td>1182</td>
      <td>DL-Valine</td>
      <td>{odorless}</td>
    </tr>
    <tr>
      <th>7615</th>
      <td>9064</td>
      <td>Cianidanol</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7616</th>
      <td>72276</td>
      <td>(-)-Epicatechin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7617</th>
      <td>65064</td>
      <td>(-)-Epigallocatechin gallate</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7618</th>
      <td>107905</td>
      <td>(-)-Epicatechin gallate</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7619</th>
      <td>5280343</td>
      <td>quercetin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7620</th>
      <td>65084</td>
      <td>Gallocatechin</td>
      <td>{bitter}</td>
    </tr>
    <tr>
      <th>7621</th>
      <td>994</td>
      <td>Dl-Phenylalanine</td>
      <td>{odorless}</td>
    </tr>
    <tr>
      <th>7622</th>
      <td>445154</td>
      <td>resveratrol</td>
      <td>{bitter}</td>
    </tr>
  </tbody>
</table>
<p>7623 rows × 3 columns</p>
</div>




```python
print('Missing IDs: ' + str(missing_ids))
```

    Missing IDs: [406, 407, 420, 479, 483, 599, 605, 666, 681, 689, 692, 760, 761, 779, 797, 798, 801, 802, 804, 808, 809, 811, 812, 813, 816, 819, 838, 844, 866, 877, 888, 892, 903, 910, 922, 940, 946, 957, 966, 973, 974, 975, 976]


The missing IDs might be due to a bad internet connection, as opposed to the content actually missing, so redownload them just to be sure.


```python
ranges = [(i-1, i) for i in missing_ids]
flavor_df, molecules_df, missing_ids = update_flavordb_dataframes(flavor_df, molecules_df, ranges)
print('# of missing IDs: ' + str(len(missing_ids)))
print('Missing IDs: ' + str(missing_ids))
```

    Downloading took: 0.8541397333145142 minutes
    # of missing IDs: 43
    Missing IDs: [405, 406, 419, 478, 482, 598, 604, 665, 680, 688, 691, 759, 760, 778, 796, 797, 800, 801, 803, 807, 808, 810, 811, 812, 815, 818, 837, 843, 865, 876, 887, 891, 902, 909, 921, 939, 945, 956, 965, 972, 973, 974, 975]


Done! Now we have a large database of foods. But how do we know if the database is complete enough? Let's test how many foods FlavorDB knows.


```python
foods = ['caramel', 'urchin', 'liver', 'haggis',
         'blood', 'cheese', 'pawpaw', 'rose',
         'durian', 'squirrel', 'kombu', 'whale',
         'white fish', 'whitefish']

# check if any food matches (or is a substring of) an alias in the database
{f : any([f in alias for alias in flavor_df['alias']])
 for f in foods}
```




    {'caramel': False,
     'urchin': False,
     'liver': False,
     'haggis': False,
     'blood': False,
     'cheese': True,
     'pawpaw': True,
     'rose': True,
     'durian': True,
     'squirrel': True,
     'kombu': True,
     'whale': True,
     'white fish': False,
     'whitefish': True}



Hmmm. This database is not exactly complete. While the database certainly includes some uncommon foods like whale, durian, pawpaw, and rose, it is also missing others such as sea urchin, liver, and blood. In addition, common terms, like "white fish", which refers to several species of fish, are left out entirely ("whitefish" refers to a single species of fish).

Of course, we wouldn't expect this database to have the food compounds of caramel, because even today, the process of caramelization is extremely complex and not well-understood, so complete information on caramel shouldn't be there.
