
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

[Food pairing](https://en.wikipedia.org/wiki/Foodpairing) is a principle for deciding which foods, when eaten together, are better flavor-wise. [One study, *Flavor network and the principles of food
pairing* published in *Nature*](https://www.nature.com/articles/srep00196), found that Western cooking, for example, prefers to pair foods when they share many flavors, as opposed to Eastern cooking, which tends to pair foods when they contrast flavor-wise. When combined with knowledge of what foods are central to each culture, it then becomes possible to derive insights what gives each cuisine its style.

Hypothetically, a [fusion restuarant](https://en.wikipedia.org/wiki/Fusion_cuisine) might choose to cook in a distinctly Eastern style (contrasting flavors) using mostly Western ingredients. It might even be possible to generate new cuisine styles that are designed to be as different as possible from existing styles. However, before any of that happens, we'll need data on food, and lots of it.

# Overview

First, we'll need a database of flavor compounds in each kind of food ingredient. Several databases concerning food exist, such as [FoodDB](http://foodb.ca/), [FlavorNet](http://www.flavornet.org/), and [FlavorDB](https://www.ncbi.nlm.nih.gov/pubmed/29059383), but not all associate foods with the compounds they contain. The one at FlavorDB does, so we can scrape our data from the FlavorDB [website](https://cosylab.iiitd.edu.in/flavordb/).

Once we have the data, we'll need it in a form that we can easily manipulate. [``Pandas DataFrames``](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html) are a good choice - they're effectively small databases that we can access and manipulate with the power of Python's other libraries, and without the need for a SQL-like syntax.

Then, we'll be able to run all sorts of nice data visualizations and analysis methods on our ``DataFrames``.

# Acquiring Data

First things first - how do we go about scraping the data from FlavorDB?

The general steps to data scraping are:
1. **Download** the JSON files which describe the chemical makeup of the food.
    - Find the URLs which have the JSON files.
    - Iterate over the URLs that have these JSON files, and download the data in them.
2. **Process** the JSON data by converting it into a Pandas ``Dataframe``.
3. **Clean** the ``DataFrame``.

## Steps 1-2: JSON Files

A quick inspection of the FlavorDB website reveals that all of the JSON files we want are at https://cosylab.iiitd.edu.in/flavordb/entities_json?id=x, where ``x`` is an integer. Then it's a cinch to write a few functions which go to those addresses and converts those JSON files into dictionaries. 


```python
# import the relevant Python packages
!pip install mpld3
!pip install "git+https://github.com/javadba/mpld3@display_fix"
!pip install colour

# for basic data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# for downloading files off the internet
import urllib.request
import json
import time


# for network graphs
from colour import Color
from matplotlib.collections import LineCollection
import networkx as nx
import mpld3
```

    Requirement already satisfied: mpld3 in /home/jovyan/.local/lib/python3.6/site-packages
    [33mYou are using pip version 9.0.3, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Collecting git+https://github.com/javadba/mpld3@display_fix
      Cloning https://github.com/javadba/mpld3 (to display_fix) to /tmp/pip-s3uo8vu6-build
      Requirement already satisfied (use --upgrade to upgrade): mpld3==0.3.1.dev1 from git+https://github.com/javadba/mpld3@display_fix in /home/jovyan/.local/lib/python3.6/site-packages
    [33mYou are using pip version 9.0.3, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Requirement already satisfied: colour in /opt/conda/lib/python3.6/site-packages
    [33mYou are using pip version 9.0.3, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
# JSON files are at addresses of this form
def flavordb_entity_url(x):
    return "https://cosylab.iiitd.edu.in/flavordb/entities_json?id="+str(x)


# translates the JSON file at the specified web address into a dictionary
def get_flavordb_entity(x):
    # source: https://stackoverflow.com/questions/12965203/how-to-get-json-from-webpage-into-python-script
    with urllib.request.urlopen(flavordb_entity_url(x)) as url:
        return json.loads(url.read().decode())
    return None
```

Those dictionaries contain a lot of unnecessary information, which is why we need to specify what fields we want (and, for ease of use, what we want to rename them to).


```python
# the names of the "columns" in the raw JSON objects
def flavordb_entity_cols():
    return [
        'entity_id', 'entity_alias_readable', 'entity_alias_synonyms',
        'natural_source_name', 'category_readable', 'molecules'
    ]


# what we want to rename the JSON object "columns" to
def flavordb_df_cols():
    return [
        'entity id', 'alias', 'synonyms',
        'scientific name', 'category', 'molecules'
    ]


# "subcolumns" in the "molecules" column that we are interested in
def molecules_df_cols():
    return ['pubchem id', 'common name', 'flavor profile']    
```

## Steps 3-4: Downloading & Cleaning

We still haven't actually wrote anything that executes - just a bunch of definitions. Right now, we need to define a bunch of stuff as setup. Then, when we download the JSON data later, we can immediately pipeline it through processing and cleaning.

The end goal is to have the data in a consistent, easy to access format. That means that I want two databases, one associating foods with food compounds, and another associating food compounds with flavors. The database columns should have the following types:
* 'entity id' and 'pubchem id' to be type ``int``
* 'alias', 'scientific name', 'category', and 'common name' to be type ``str``
* 'synonyms' and 'flavor profile' should be type ``set(str)``
* 'molecules' should be a ``set(int)``

When these columns are initially downloaded, some of them, such as 'scientific name', have mixed types - they have both strings and some other value (in this case, ``NaN``). Fortunately, not too many columns have mixed types. In particular, (if you do some preliminary work and examine the JSON files,) the columns 'entity id', 'pubchem id', and 'common name' don't require type checking, so we don't have to process those at all.

The other small thing that do here is call ``str.lower()`` on all of the strings, just so it's easier to type and it can match other recipes more easily.


```python
def clean_flavordb_dataframes(flavor_df, molecules_df):
    """
    Helps ensure consistent intra-column typing and converts all strings to lowercase.
    """
    strtype = type('')
    settype = type(set())
    
    # ensuring that these columns have type str
    for k in ['alias', 'scientific name', 'category']:
        flavor_df[k] = [
            elem.strip().lower() if isinstance(elem, strtype) else ''
            for elem in flavor_df[k]
        ]
    
    # ensuring that these columns are always a set of str
    def map_to_synonyms_set(elem):
        if isinstance(elem, settype):
            return elem
        elif isinstance(elem, strtype):
            # if it's a string of a set,
            if elem[0] == '{' and elem[-1] == '}':
                # convert it to a set
                return eval(elem)
            else:
                # else it's probably directly from source
                return set(elem.strip().lower().split(', '))
        else:
            return set()
    
    flavor_df['synonyms'] = [
        map_to_synonyms_set(elem)
        for elem in flavor_df['synonyms']
    ]
    
    molecules_df['flavor profile'] = [
        set([x.strip().lower() for x in elem])
        for elem in molecules_df['flavor profile']
    ]
    
    return [
        flavor_df.groupby('entity id').first().reset_index(),
        molecules_df.groupby('pubchem id').first().reset_index()
    ]
```

This is where most of the work is done. ``get_flavordb_dataframes()`` is the code that ties together all three steps of data scraping: **downloading**, **processing**, and **cleaning**. It even handles errors, for when a JSON page is missing.


```python
# generate dataframes from some of the JSON objects
def get_flavordb_dataframes(start, end):
    """
    Download JSON data, converts it to DataFrames, and cleans them.
    
    Returns DataFrames for both foods and molecules, as well as missing JSON entries.
    """
    # make intermediate values to make dataframes from
    flavordb_data = []
    molecules_dict = {}
    missing = [] # numbers of the missing JSON files during iteration
    
    flavordb_cols = flavordb_entity_cols()
    
    for i in range(start, end):
        # we use a try-except here because some of the JSON pages are missing
        try:
            # 1: Find the JSON file. Gets the ith food entity, as a JSON dict
            fdbe = get_flavordb_entity(i + 1)

            # get only the relevant fields (columns) of the dict
            flavordb_series = [fdbe[k] for k in flavordb_cols[:-1]]
            flavordb_series.append( # convert the field to a set
                set([m['pubchem_id'] for m in fdbe['molecules']])
            )
            flavordb_data.append(flavordb_series)

            # update the molecules dataframe with the data in 'molecules' field
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

It takes a long time to download all of these JSON files. If the code somehow crashes, we'll lose all of our download progress in a few short seconds. Therefore, it's a good idea to save the download progress.


```python
# updates & saves the download progress of your dataframes
def update_flavordb_dataframes(df0, df1, ranges):
    """
    Adds more data to the specified DataFrames, and saves them as CSV files.
    
    If successful, returns the specified DataFrames, now updated, and any missing JSON files.
    """
    df0_old = df0
    df1_old = df1
    missing_old = []

    # time how long it took to download the files
    start = time.time()
    
    # for each range in ranges, save your progress.
    # don't continue with the program unless everything succeeds!
    try:
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

As of today, it looks like FlavorDB has about 1,000 distinct foods (entities). We'll get the first 1,000 foods we find, and save our progress about every 50 or so foods downloaded.


```python
# take new dataframes
df0 = pd.DataFrame(columns=flavordb_df_cols())
df1 = pd.DataFrame(columns=molecules_df_cols())

# fill the DataFrames with JSON files up to id = 1000
ranges = [(50 * i, 50 * (i + 1)) for i in range(20)]
# update & save the dataframes as csv files
update_flavordb_dataframes(df0, df1, ranges)
```

Creating a DataFrame from a CSV file is a lot faster than downloading and creating one from the internet. In a perfect world, we wouldn't need to, but in the interest of saving time, I've made these methods so that I don't need to redownload the ``DataFrames`` every time I make an edit to the code. They load the ``DataFrame``s from CSV files and recover the information about what JSON IDs are missing.


```python
# get the missing entries
def missing_entity_ids(flavor_df):
    """
    Get the IDs of the missing JSON entries for this particular food DataFrame.
    """
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

Okay, now we can finally display a few rows of our ``DataFrame``s.


```python
# missing_ids = the missing ids that are less than the max one found
flavor_df, molecules_df, missing_ids = load_db()
flavor_df.to_csv('flavordb.csv')
molecules_df.to_csv('molecules.csv')
flavor_df.head()
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
      <td>{7361, 994, 10883, 7362, 11173, 5365891, 11559...</td>
    </tr>
  </tbody>
</table>
</div>




```python
molecules_df.head()
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
      <td>4</td>
      <td>1-Aminopropan-2-ol</td>
      <td>{fishy}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>3-Methyl-2-oxobutanoic acid</td>
      <td>{fruity}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>2-oxobutanoic acid</td>
      <td>{sweet, creamy, caramel, lactonic, brown}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>4-Methyl-2-oxovaleric acid</td>
      <td>{fruity}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>3,4-Dihydroxybenzoic Acid</td>
      <td>{mild, balsamic, phenolic}</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Missing IDs: ' + str(missing_ids))
```

    Missing IDs: [406, 407, 420, 479, 483, 599, 605, 666, 681, 689, 692, 760, 761, 779, 797, 798, 801, 802, 804, 808, 809, 811, 812, 813, 816, 819, 838, 844, 866, 877, 888, 892, 903, 910, 922, 940, 946, 957, 966, 973, 974, 975, 976]


One last check: what are the values in category?


```python
str(set(flavor_df['category']))
```




    "{'spice', 'beverage caffeinated', 'seed', 'additive', 'fruit', 'vegetable', 'herb', 'vegetable root', 'vegetable tuber', 'vegetable fruit', 'legume', 'meat', 'vegetable stem', 'beverage', 'maize', 'berry', 'cabbage', 'beverage alcoholic', 'plant', 'fish', 'dish', 'essential oil', 'plant derivative', 'fungus', 'seafood', 'bakery', 'fruit citrus', 'cereal', 'gourd', 'dairy', 'nut', 'fruit-berry', 'flower', 'fruit essence'}"



Oops. It looks like we still have some more data cleaning to do. What's the difference between a vegetable, fruit, and vegetable fruit? How come cabbage gets its own category? Let's take a closer look. If we take a quick glance through the names of *every* food in FlavorDB (output not included), we'll notice a few strange things:


```python
aliases_by_category = ''
for c in set(flavor_df['category']):
    aliases_by_category += (
        c + ': '
        + str(list(flavor_df[flavor_df['category'] == c]['alias']))
        + '\n\n'
    )
# check out the output of this yourself, if you like
# print(aliases_by_category)
```

It looks like some entries/categories were made erroneously (see elderberry, cornbread, japanese pumpkin). A few looked incorrect but were correct (corn salad, or cornsalad, is a type of leafy vegetable), but a lot were seemed like they were sorted to make sure no one category is too large. However, I can see signficant differences in flavor between categories.

We're mostly interested in the ingredients list, not in finished products like cornbread, so we'll keep only raw ingredients, remove a few outliers, and give them each a food group:

(Also: woah! [Apparently tobacco is a food.](https://www.scmp.com/magazines/post-magazine/article/1701428/how-cook-using-tobacco-sweet-and-savoury-dishes))


```python
def food_groups():
    return set([
        'grain', 'vegetable', 'fruit', 'protein',
        'dairy', 'fat', 'sugar', 'seasoning',
        'beverage', 'alcohol'
    ])


# don't worry about the details in this! It's just a lot of sorting.
def get_food_group(food, category):
    """
    Maps each food category to a food group.
    
    The food groups include the main five: grain, vegetable, fruit, protein,
    dairy, fat, and sugar. However, they also include others: seasonings,
    beverages, alcohol.
    """
    
    out = None # return None if you don't know/want to classify it
    
    # broadly classify the major food groups
    if category in ['bakery', 'vegetable tuber', 'cereal']:
        out = 'grain'
    elif category in [
        'flower', 'fungus', 'plant', 'cabbage',
        'vegetable fruit', 'herb', 'gourd', 'vegetable'
    ]:
        out = 'vegetable'
    elif category in [
        'fruit-berry', 'berry', 'fruit', 'fruit citrus'
    ]:
        out = 'fruit'
    elif category in [
        'legume', 'nut', 'seed', 'seafood', 'fish', 'meat'
    ]:
        out = 'protein'
    elif category in ['dairy']:
        out = 'dairy'
    elif category in [
        'fruit essence', 'additive', 'spice', 'essential oil'
    ]:
        out = 'seasoning'
    elif category in ['beverage alcoholic']:
        out = 'alcohol'
    elif 'beverage' in category:
        out = 'beverage'
    elif category == 'maize':
        if food in ['corn', 'sweetcorn']:
            out = 'vegetable'
        elif food in ['cornbread', 'corn grits', 'popcorn']:
            out = 'grain'
        elif food == 'corn oil':
            out = 'fat'
    elif category == 'plant derivative':
        if (any(x in food for x in ['sauce', 'vinegar'])
            or food in ['creosote', 'storax', 'cocoa powder']):
            # creosote is what gives smoky foods that smoky flavor
            # storax is...weird
            out = 'seasoning'
        elif 'seed' in food or food == 'peanut butter':
            # cottonseeds are now available for people to eat!
            out = 'protein'
        elif any([x in food for x in ['butter', 'oil']]):
            out = 'fat'
        elif food == 'fermented tea':
            out = 'beverage'
        elif food in ['honey', 'chocolate', 'chocolate spread']:
            out = 'sugar'
        elif food == 'macaroni':
            out = 'grain'
        elif food in ['jute', 'tofu']:
            out = 'vegetable'
        elif food == 'soy yogurt':
            out = 'dairy'
    elif category == 'additive':
        if 'sugar' in food or food in [
            'fruit preserve', 'syrup', 'icing', 'molasses'
        ]:
            out = 'sugar'
        elif 'margarine' in food or food in ['cooking oil', 'shortening']:
            out = 'fat'
        elif food in ['sauce', 'gelatin dessert', 'spread', 'topping', 'water']:
            out = None # don't know how to classify these items
        elif food == 'stuffing':
            out = 'grain'
        else:
            out = 'seasoning'
    
    # cover exceptions to the rule
    if (
        any([
            food == x + ' oil'
            for x in ['soybean', 'cooking', 'fish', 'peanut', 'canola', 'corn']
        ])
        or food in ['butter', 'ghee']
        or (' butter' in food and food != 'peanut butter')
        or 'margarine' in food
    ):
        out = 'fat'
    elif food in [
        'sugar', 'honey', 'molasses', 'agave', 'dulce de leche'
    ]:
        # these were classified under 'additives/dairy/plant derivative'
        out = 'sugar'
    elif food in ['irish moss', 'kelp', 'kombu', 'wakame']:
        # these were classified under 'seafood'
        out = 'vegetable'
    elif food in ['butternut squash', 'winter squash', 'japanese pumpkin']:
        # these were classified under 'fruit'
        out = 'vegetable'
    elif food in ['sweet custard', 'candy bar', 'chocolate mousse', 'fudge']:
        out = 'sugar'

    return out


# make a DataFrame saving the results & food groups
ridf = flavor_df.copy() # ridf = raw ingredients df
ridf['group'] = [
    get_food_group(ridf.at[i, 'alias'], ridf.at[i, 'category'])
    for i in ridf.index
]
ridf = ridf[[
    g is not None
    for g in ridf['group']
]]
ridf = ridf.reset_index()
ridf.head()
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
      <th>index</th>
      <th>entity id</th>
      <th>alias</th>
      <th>synonyms</th>
      <th>scientific name</th>
      <th>category</th>
      <th>molecules</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>bakery products</td>
      <td>{bakery products}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{27457, 7976, 31252, 26808, 22201, 26331}</td>
      <td>grain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>bread</td>
      <td>{bread}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{1031, 1032, 644104, 527, 8723, 31260, 15394, ...</td>
      <td>grain</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>rye bread</td>
      <td>{rye bread}</td>
      <td>rye</td>
      <td>bakery</td>
      <td>{644104, 7824, 643731, 8468, 1049, 5372954, 80...</td>
      <td>grain</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>wheaten bread</td>
      <td>{soda scones, soda farls}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{6915, 5365891, 12170, 8082, 31251, 7958, 1049...</td>
      <td>grain</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>white bread</td>
      <td>{white bread}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{7361, 994, 10883, 7362, 11173, 5365891, 11559...</td>
      <td>grain</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory Data Analysis

## Preliminary Analysis

Done! Now we have a large database of foods. But how comprehensive is the data? Let's check by looking for some odd foods.


```python
foods = ['caramel', 'urchin', 'liver',
         'blood', 'cheese', 'pawpaw', 'rose',
         'durian', 'squirrel', 'kombu', 'whale',
         'white fish', 'whitefish']

# check if any food matches (or is a substring of) an alias in the database
{f : any([f in alias for alias in ridf['alias']])
 for f in foods}
```




    {'caramel': False,
     'urchin': False,
     'liver': False,
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



Hmmm. This database is not exactly complete. While the database certainly includes some uncommon foods like [whale](https://en.wikipedia.org/wiki/Whale_meat), [durian](https://en.wikipedia.org/wiki/Durian), [paw-paw](https://en.wikipedia.org/wiki/Asimina_triloba), and [rose](https://en.wikipedia.org/wiki/Rose#Food_and_drink), it is also missing others such as [sea urchin](https://en.wikipedia.org/wiki/Sea_urchin#As_food), [liver](https://en.wikipedia.org/wiki/Liver_(food)), and [blood](https://en.wikipedia.org/wiki/Blood_as_food) (see [black pudding](https://en.wikipedia.org/wiki/Black_pudding)). In addition, common terms, like ["white fish"](https://en.wikipedia.org/wiki/Whitefish_(fisheries_term)), which refers to several species of fish, are left out entirely ("whitefish" refers to a single species of fish).

Of course, we wouldn't expect this database to have the food compounds of caramel, because even today, the [process of caramelization](https://www.scienceofcooking.com/caramelization.htm) is [extremely complex](https://www.exploratorium.edu/cooking/candy/caramels-story.html) and [not well-understood](https://bcachemistry.wordpress.com/2014/05/11/the-chemistry-of-caramel/), so [complete information on caramel](https://chem-net.blogspot.com/2015/04/food-chemistry-caramelization-sugar15.html) shouldn't be there.

Now that's out of the way, it's time for analysis and visualizations!

## Similar Foods

Which foods are most similar to each other? From the previously mentioned [Nature article](https://www.nature.com/articles/srep00196), the mean number of shared compounds per recipe is given by ``msc()``:


```python
def get_food(food_name, fdf):
    return fdf[[
        (fdf.at[i, 'alias'] == food_name
         or food_name in fdf.at[i, 'synonyms'])
        for i in fdf.index
    ]]


def get_molecules(food_name, fdf):
    out = list(get_food(food_name, fdf)['molecules'])
    if len(out) > 1:
        raise ValueError('food ' + food_name + ' has more than one entry')
    return out[0]
    
    
def msc(foods, fdf, **kwargs):
    """
    Return the mean shared compounds (MSC) for a given recipe (set of foods),
    i.e. sum(# shared compounds per 2 foods) / (# of combinations of 2 foods)
    """
    use_index = kwargs.get('use_index', False)
    if use_index:
        mols = [fdf.at[i, 'molecules'] for i in foods]
    else:
        mols = [get_molecules(f, fdf) for f in foods]
    
    nr = len(foods)
    out = 0
    for i in range(nr):
        for j in range(i + 1, nr):
            out += len(mols[i].intersection(mols[j]))
    out *= 2.0 / (nr * (nr - 1))
    return out
```

Since we only have ~1,000 foods in our ``DataFrame``, it's not too expensive to find the MSC between every two foods. Then, we can look for "clusters" of similar foods and ones that very different from one another.


```python
def block_msc(foods0, foods1, fdf, **kwargs):
    """
    Get the MSC when comparing each of the foods in foods0 to each of the foods in foods1.
    """
    len0 = len(foods0)
    len1 = len(foods1)
    
    out = np.ndarray((len0, len1))
    for i in range(len0):
        for j in range(len1):
            out[i][j] = msc([foods0[i], foods1[j]], fdf, **kwargs)
    
    return out


def intragroup_msc(foods, fdf, **kwargs):
    lenf = len(foods)
    food_msc = block_msc(foods, foods, fdf, **kwargs)
    
    out = []
    for i in range(lenf):
        out.append([])
        for j in range(1 + i, lenf):
            out[-1].append(food_msc[i][j])
    
    return out


def flatten(ls):
    return [x for sublist in ls for x in sublist]
```


```python
msc_data = flatten(intragroup_msc(ridf.index, ridf, use_index=True))
```

Now that we have the MSC between all pairs of food, let's see how many compounds foods normally share:


```python
print('Average: ' + str(np.average(msc_data)))
print('Median: ' + str(np.median(msc_data)))

fignum = 1
plt.hist(msc_data, bins=list(range(60)))
plt.title('Figure ' + str(fignum) + ':\nFrequency of Mean Shared Compounds')
plt.xlabel('mean shared compounds')
plt.ylabel('frequency')
plt.show()
fignum += 1
```

    Average: 21.7043497344
    Median: 3.0



![png](food-tutorial_files/food-tutorial_34_1.png)


This shouldn't be that surprising; only similar foods (like beef and pork) should have similar compounds in them. The vast majority of foods taste really different from one another!

But wait, how similar are foods inside and outside their own food groups? To visualize that, use a violin plot:


```python
def make_violin_plot(x, y, **kwargs):
    xl = kwargs.get('x_label', '')
    yl = kwargs.get('y_label', '')
    t = kwargs.get('title', '')
    w = kwargs.get('widths', 4)
    fs = kwargs.get('figsize', (10, 7.5))
    
    # create a sufficiently wide violin plot, with a median
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fs)
    ax.violinplot(y, x, widths=w, showmedians=True)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(t)
```

Huh. Looks like there are small groups of fats, vegetables, proteins, and dairy products that taste similar within themselves. Interesting.


```python
food_grouped = ridf.groupby(by='group')

x = list(food_groups())
xtic = list(range(len(x)))
y = [
    flatten(intragroup_msc(list(group.index), ridf, use_index=True))
    for fg, group in food_grouped
]
make_violin_plot(
    xtic,
    y,
    widths=0.9,
    x_label='food group',
    y_label='intragroup MSC'
)
plt.xticks(xtic, x)
plt.title('Figure ' + str(fignum) + ':\nDistribution of Intra-group MSC across Food Groups')
plt.show()
fignum += 1
```


![png](food-tutorial_files/food-tutorial_38_0.png)


We should also double check if our results (the number of shared compounds) match the [existing literature has reported](https://www.nature.com/articles/srep00196):


```python
# from Figure 3 of the Nature article,
expected_num_compounds = [
    {'lemon': 69, 'shrimp': 63, 'shared': 9},
    {'coffee': 132, 'beef': 97, 'shared': 102}
]

print('Expected # of compounds:')
for x in expected_num_compounds:
    print(x)
print('')

print('Actual # of compounds:')
actual_num_compounds = []
for f0, f1, s in expected_num_compounds:
    # food 1, food 2, shared
    # get the molecules in each food
    mols = [get_molecules(x, ridf) for x in [f0, f1]]
    mols.append(mols[0].intersection(mols[1]))
    mols = [len(x) for x in mols]
    print({k: v for k, v in zip([f0, f1, s], mols)})
```

    Expected # of compounds:
    {'lemon': 69, 'shrimp': 63, 'shared': 9}
    {'coffee': 132, 'beef': 97, 'shared': 102}
    
    Actual # of compounds:
    {'lemon': 193, 'shrimp': 76, 'shared': 17}
    {'coffee': 269, 'beef': 92, 'shared': 37}


Looks like our database (FlavorDB) is somewhat different from the one that they used (which was derived from *Fenaroli's Handbook of Flavor Ingredients*). However, we might still be able to get useful information out of an analysis.


```python
msc_matrix = block_msc(ridf.index, ridf.index, ridf, use_index=True)
ridf['msc sum'] = [
    sum(msc_matrix[i]) - msc_matrix[i][i]
    for i in ridf.index
]
ridf.head()
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
      <th>index</th>
      <th>entity id</th>
      <th>alias</th>
      <th>synonyms</th>
      <th>scientific name</th>
      <th>category</th>
      <th>molecules</th>
      <th>group</th>
      <th>msc sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>bakery products</td>
      <td>{bakery products}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{27457, 7976, 31252, 26808, 22201, 26331}</td>
      <td>grain</td>
      <td>261.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>bread</td>
      <td>{bread}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{1031, 1032, 644104, 527, 8723, 31260, 15394, ...</td>
      <td>grain</td>
      <td>39386.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>rye bread</td>
      <td>{rye bread}</td>
      <td>rye</td>
      <td>bakery</td>
      <td>{644104, 7824, 643731, 8468, 1049, 5372954, 80...</td>
      <td>grain</td>
      <td>5497.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>wheaten bread</td>
      <td>{soda scones, soda farls}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{6915, 5365891, 12170, 8082, 31251, 7958, 1049...</td>
      <td>grain</td>
      <td>1017.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>white bread</td>
      <td>{white bread}</td>
      <td>wheat</td>
      <td>bakery</td>
      <td>{7361, 994, 10883, 7362, 11173, 5365891, 11559...</td>
      <td>grain</td>
      <td>748.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def make_network(edge_df, source, dest, **kwargs):
    """
    Make a network graph with labels.
    """
    nodelist = kwargs.get('nodelist', None)
    node_sizes = kwargs.get('node_sizes', None)
    node_colors = kwargs.get('node_colors', None)
    node_tooltips = kwargs.get('node_tooltips', None)
    lw = kwargs.get('linewidths', len(nodelist) * [0.25])
    fs = kwargs.get('figsize', (12,  8))
    t = kwargs.get('title', '')
    
    # associate each list with a key
    node_arrs = {
        'size': node_sizes,
        'color': node_colors,
        'tooltip': node_tooltips
    }
    
    # associate each node with some data
    node_data = {
        nodelist[i]: {
            k: arr[i]
            for k, arr in node_arrs.items()
            if arr is not None
        }
        for i in range(len(nodelist))
    }
    
    # create graph from edge list
    G = nx.from_pandas_edgelist(edge_df, source, dest)
    # generate positions of nodes in graph
    pos = nx.spring_layout(G)
    nodes, x, y = zip(*[[k, v[0], v[1]] for k, v in pos.items()])
    
    # now that we have a different order of nodes, change the order
    node_arrs = {
        k: None if arr is None else [node_data[n][k] for n in nodes]
        for k, arr in node_arrs.items()
    }
    
    # prepare the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fs)
    # add the nodes
    scatter = ax.scatter(x, y, s=node_arrs['size'], c=node_arrs['color'])
    
    # add the edges
    line_segments = LineCollection(
        [
            [pos[src], pos[dst]]
            for src, dst in zip(edge_df[source], edge_df[dest])
        ],
        colors='black',
        linewidths=lw
    )
    ax.add_collection(line_segments)
    
    # add tooltips, if any
    if node_arrs['tooltip'] is not None:
        tooltip = mpld3.plugins.PointLabelTooltip(
            scatter,
            labels=node_arrs['tooltip']
        )
        mpld3.plugins.connect(fig, tooltip)
    else:
        mpld3.plugins.connect(fig)
    
    # some extra style to help with the graph
    ax.grid(color='white', linestyle='solid')
    plt.title(t)
    # make sure to follow up with mpld3.display() )
```


```python
def trubetskoy_colors():
    """
    Return 20 distinct colors as detailed by Sasha Trubetskoy.
    
    https://sashat.me/tag/color/
    """
    return [
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#9A6324", "#42d4f4", "#f032e6", "#bfef45",
        "#fabebe", "#469990", "#e6beff", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"
    ]
    

def get_color_gradient(color0, color1, steps):
    """
    Get hex color values in a gradient, from color0 to color1.
    """
    return [c.rgb for c in list(color0.range_to(color1, steps))]
```


```python
ridf100 = ridf.sort_values(by='msc sum', axis=0, ascending=False).head(100)
ridf100
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
      <th>index</th>
      <th>entity id</th>
      <th>alias</th>
      <th>synonyms</th>
      <th>scientific name</th>
      <th>category</th>
      <th>molecules</th>
      <th>group</th>
      <th>msc sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>308</th>
      <td>309</td>
      <td>310</td>
      <td>tea</td>
      <td>{tea}</td>
      <td>camellia sinensis</td>
      <td>plant</td>
      <td>{12297, 5281804, 77837, 522266, 4133, 6184, 53...</td>
      <td>vegetable</td>
      <td>47728.0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>161</td>
      <td>162</td>
      <td>apple</td>
      <td>{apple}</td>
      <td>malus</td>
      <td>fruit</td>
      <td>{8193, 8194, 229377, 12293, 12294, 1031, 64410...</td>
      <td>fruit</td>
      <td>45932.0</td>
    </tr>
    <tr>
      <th>182</th>
      <td>182</td>
      <td>183</td>
      <td>guava</td>
      <td>{guava}</td>
      <td>psidium guajava</td>
      <td>fruit</td>
      <td>{31234, 644104, 429065, 4564493, 12813, 535246...</td>
      <td>fruit</td>
      <td>45031.0</td>
    </tr>
    <tr>
      <th>359</th>
      <td>363</td>
      <td>364</td>
      <td>tomato</td>
      <td>{tomato}</td>
      <td>solanum</td>
      <td>vegetable fruit</td>
      <td>{5283335, 644104, 1032, 1031, 5283339, 527, 53...</td>
      <td>vegetable</td>
      <td>44776.0</td>
    </tr>
    <tr>
      <th>181</th>
      <td>181</td>
      <td>182</td>
      <td>grape</td>
      <td>{grape}</td>
      <td>vitis</td>
      <td>fruit</td>
      <td>{5283329, 61953, 62465, 5364231, 644104, 42906...</td>
      <td>fruit</td>
      <td>44663.0</td>
    </tr>
    <tr>
      <th>189</th>
      <td>189</td>
      <td>190</td>
      <td>mango</td>
      <td>{mango}</td>
      <td>mangifera</td>
      <td>fruit</td>
      <td>{8193, 8194, 5283335, 644104, 429065, 12810, 1...</td>
      <td>fruit</td>
      <td>44377.0</td>
    </tr>
    <tr>
      <th>281</th>
      <td>282</td>
      <td>283</td>
      <td>cocoa</td>
      <td>{cocoa}</td>
      <td>theobroma</td>
      <td>seed</td>
      <td>{5283329, 1030, 30215, 644104, 429065, 1031, 1...</td>
      <td>protein</td>
      <td>44141.0</td>
    </tr>
    <tr>
      <th>363</th>
      <td>372</td>
      <td>373</td>
      <td>potato</td>
      <td>{potato}</td>
      <td>solanum</td>
      <td>vegetable tuber</td>
      <td>{8193, 30215, 644104, 31246, 527, 5364752, 528...</td>
      <td>grain</td>
      <td>43733.0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>195</td>
      <td>196</td>
      <td>papaya</td>
      <td>{pawpaw, papaw}</td>
      <td>carica</td>
      <td>fruit</td>
      <td>{12294, 1031, 644104, 429065, 1032, 31246, 527...</td>
      <td>fruit</td>
      <td>43684.0</td>
    </tr>
    <tr>
      <th>233</th>
      <td>233</td>
      <td>234</td>
      <td>strawberry</td>
      <td>{strawberry}</td>
      <td>fragaria</td>
      <td>berry</td>
      <td>{31234, 1031, 644104, 429065, 12810, 229387, 5...</td>
      <td>fruit</td>
      <td>43640.0</td>
    </tr>
    <tr>
      <th>287</th>
      <td>288</td>
      <td>289</td>
      <td>soybean</td>
      <td>{soya bean }</td>
      <td>glycine</td>
      <td>legume</td>
      <td>{5283335, 644104, 1032, 1031, 5283339, 527, 52...</td>
      <td>protein</td>
      <td>43609.0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>245</td>
      <td>246</td>
      <td>mushroom</td>
      <td>{mushroom}</td>
      <td>agaricus bisporus</td>
      <td>fungus</td>
      <td>{8193, 31234, 20995, 1030, 1031, 644104, 1032,...</td>
      <td>vegetable</td>
      <td>43253.0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>173</td>
      <td>174</td>
      <td>black currant</td>
      <td>{black currant}</td>
      <td>ribes</td>
      <td>fruit</td>
      <td>{644104, 1032, 31244, 527, 7695, 8723, 31253, ...</td>
      <td>fruit</td>
      <td>43226.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>193</td>
      <td>194</td>
      <td>orange</td>
      <td>{orange}</td>
      <td>citrus</td>
      <td>fruit</td>
      <td>{8194, 12294, 1031, 644104, 1032, 527, 5364752...</td>
      <td>fruit</td>
      <td>43140.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>54</td>
      <td>55</td>
      <td>rice</td>
      <td>{rice}</td>
      <td>oryza sativa</td>
      <td>cereal</td>
      <td>{5283329, 8193, 5283335, 644104, 1031, 1032, 5...</td>
      <td>grain</td>
      <td>43067.0</td>
    </tr>
    <tr>
      <th>163</th>
      <td>163</td>
      <td>164</td>
      <td>apricot</td>
      <td>{armenia,  zardalu}</td>
      <td>prunus</td>
      <td>fruit</td>
      <td>{12293, 12294, 1031, 644104, 12810, 12813, 527...</td>
      <td>fruit</td>
      <td>42988.0</td>
    </tr>
    <tr>
      <th>331</th>
      <td>332</td>
      <td>333</td>
      <td>ginger</td>
      <td>{ginger}</td>
      <td>zingiber</td>
      <td>spice</td>
      <td>{5356544, 220674, 8194, 1031, 644104, 527, 442...</td>
      <td>seasoning</td>
      <td>42984.0</td>
    </tr>
    <tr>
      <th>340</th>
      <td>341</td>
      <td>342</td>
      <td>green beans</td>
      <td>{french beans, snap beans, string beans}</td>
      <td>phaseolus vulgaris</td>
      <td>vegetable</td>
      <td>{8193, 5283335, 644104, 1031, 5283339, 527, 87...</td>
      <td>vegetable</td>
      <td>42886.0</td>
    </tr>
    <tr>
      <th>205</th>
      <td>205</td>
      <td>206</td>
      <td>pineapple</td>
      <td>{ananas}</td>
      <td>ananas</td>
      <td>fruit</td>
      <td>{8193, 1031, 644104, 1032, 429065, 85519, 527,...</td>
      <td>fruit</td>
      <td>42877.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>197</td>
      <td>198</td>
      <td>passionfruit</td>
      <td>{passionfruit}</td>
      <td>passiflora</td>
      <td>fruit</td>
      <td>{31234, 12293, 1031, 644104, 5352973, 527, 872...</td>
      <td>fruit</td>
      <td>42791.0</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>201</td>
      <td>peach</td>
      <td>{nectarine}</td>
      <td>prunus</td>
      <td>fruit</td>
      <td>{5283335, 644104, 12810, 12813, 5352973, 61455...</td>
      <td>fruit</td>
      <td>42376.0</td>
    </tr>
    <tr>
      <th>282</th>
      <td>283</td>
      <td>284</td>
      <td>beans</td>
      <td>{beans}</td>
      <td>fabaceae</td>
      <td>legume</td>
      <td>{8193, 5283335, 644104, 1031, 5283339, 527, 87...</td>
      <td>protein</td>
      <td>42215.0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>166</td>
      <td>167</td>
      <td>banana</td>
      <td>{banana}</td>
      <td>musa</td>
      <td>fruit</td>
      <td>{8193, 12294, 1031, 644104, 1032, 61455, 527, ...</td>
      <td>fruit</td>
      <td>42207.0</td>
    </tr>
    <tr>
      <th>357</th>
      <td>361</td>
      <td>362</td>
      <td>capsicum</td>
      <td>{sweet pepper}</td>
      <td>capsicum annuum</td>
      <td>vegetable fruit</td>
      <td>{62465, 6660, 644104, 12297, 527, 61455, 8723,...</td>
      <td>vegetable</td>
      <td>42129.0</td>
    </tr>
    <tr>
      <th>206</th>
      <td>206</td>
      <td>207</td>
      <td>plum</td>
      <td>{plum}</td>
      <td>prunus</td>
      <td>fruit</td>
      <td>{1031, 644104, 429065, 5281804, 5352461, 12813...</td>
      <td>fruit</td>
      <td>41997.0</td>
    </tr>
    <tr>
      <th>283</th>
      <td>284</td>
      <td>285</td>
      <td>lima beans</td>
      <td>{lima bean, sieva bean, butter bean}</td>
      <td>phaseolus lunatus</td>
      <td>legume</td>
      <td>{8193, 5283335, 644104, 1031, 5283339, 527, 87...</td>
      <td>protein</td>
      <td>41660.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>55</td>
      <td>56</td>
      <td>corn</td>
      <td>{corn}</td>
      <td>zea</td>
      <td>maize</td>
      <td>{11265, 62465, 644104, 12297, 31242, 527, 4114...</td>
      <td>vegetable</td>
      <td>41641.0</td>
    </tr>
    <tr>
      <th>180</th>
      <td>180</td>
      <td>181</td>
      <td>fig</td>
      <td>{fig}</td>
      <td>ficus carica</td>
      <td>fruit</td>
      <td>{644104, 527, 4114, 8723, 31253, 31260, 31265,...</td>
      <td>fruit</td>
      <td>41627.0</td>
    </tr>
    <tr>
      <th>190</th>
      <td>190</td>
      <td>191</td>
      <td>melon</td>
      <td>{melon}</td>
      <td>cucurbitaceae</td>
      <td>fruit</td>
      <td>{8193, 5283335, 644104, 85519, 527, 8723, 7375...</td>
      <td>fruit</td>
      <td>41598.0</td>
    </tr>
    <tr>
      <th>285</th>
      <td>286</td>
      <td>287</td>
      <td>peanut</td>
      <td>{goober, groundnut}</td>
      <td>arachis</td>
      <td>nut</td>
      <td>{5283329, 8194, 8193, 644104, 1032, 5283339, 3...</td>
      <td>protein</td>
      <td>41575.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>178</th>
      <td>178</td>
      <td>179</td>
      <td>elderberry</td>
      <td>{elder}</td>
      <td>sambucus nigra</td>
      <td>fruit-berry</td>
      <td>{5634, 644104, 527, 8723, 31251, 31253, 31260,...</td>
      <td>fruit</td>
      <td>39696.0</td>
    </tr>
    <tr>
      <th>222</th>
      <td>222</td>
      <td>223</td>
      <td>blueberry</td>
      <td>{cherry}</td>
      <td>vaccinium</td>
      <td>berry</td>
      <td>{31234, 644104, 12813, 5352461, 527, 8723, 312...</td>
      <td>fruit</td>
      <td>39695.0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>101</td>
      <td>102</td>
      <td>chamomile</td>
      <td>{english chamomile, camomile, garden chamomile...</td>
      <td>chamaemelum</td>
      <td>essential oil</td>
      <td>{644104, 527, 8723, 31253, 522266, 31260, 1254...</td>
      <td>seasoning</td>
      <td>39682.0</td>
    </tr>
    <tr>
      <th>209</th>
      <td>209</td>
      <td>210</td>
      <td>quince</td>
      <td>{quince}</td>
      <td>cydonia</td>
      <td>fruit</td>
      <td>{61953, 1031, 644104, 527, 8723, 7193, 31260, ...</td>
      <td>fruit</td>
      <td>39544.0</td>
    </tr>
    <tr>
      <th>164</th>
      <td>164</td>
      <td>165</td>
      <td>avocado</td>
      <td>{avocado pear, alligator pear}</td>
      <td>persea</td>
      <td>fruit</td>
      <td>{644104, 527, 8723, 1049, 31260, 7710, 15394, ...</td>
      <td>fruit</td>
      <td>39528.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>bread</td>
      <td>{bread}</td>
      <td>poacceae</td>
      <td>bakery</td>
      <td>{1031, 1032, 644104, 527, 8723, 31260, 15394, ...</td>
      <td>grain</td>
      <td>39386.0</td>
    </tr>
    <tr>
      <th>326</th>
      <td>327</td>
      <td>328</td>
      <td>cassia</td>
      <td>{chinese cassia, chinese cinnamon}</td>
      <td>cinnamomum</td>
      <td>spice</td>
      <td>{62465, 644104, 527, 8723, 31253, 5144, 7194, ...</td>
      <td>seasoning</td>
      <td>39360.0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>115</td>
      <td>116</td>
      <td>lovage</td>
      <td>{lovage}</td>
      <td>levisticum officinale</td>
      <td>essential oil</td>
      <td>{8194, 644104, 5352461, 527, 4114, 8723, 31253...</td>
      <td>seasoning</td>
      <td>39335.0</td>
    </tr>
    <tr>
      <th>344</th>
      <td>345</td>
      <td>346</td>
      <td>lettuce</td>
      <td>{lettuce}</td>
      <td>lettuce</td>
      <td>vegetable</td>
      <td>{644104, 5352460, 61455, 527, 8723, 31260, 153...</td>
      <td>vegetable</td>
      <td>39318.0</td>
    </tr>
    <tr>
      <th>256</th>
      <td>256</td>
      <td>257</td>
      <td>fennel</td>
      <td>{fennel}</td>
      <td>foeniculum vulgare</td>
      <td>herb</td>
      <td>{644104, 31244, 527, 4114, 8723, 31253, 537295...</td>
      <td>vegetable</td>
      <td>39231.0</td>
    </tr>
    <tr>
      <th>355</th>
      <td>358</td>
      <td>359</td>
      <td>kohlrabi</td>
      <td>{german turnip, turnip cabbage}</td>
      <td>brassica oleracea</td>
      <td>cabbage</td>
      <td>{30215, 644104, 527, 77840, 8723, 31252, 1049,...</td>
      <td>vegetable</td>
      <td>39030.0</td>
    </tr>
    <tr>
      <th>295</th>
      <td>296</td>
      <td>297</td>
      <td>walnut</td>
      <td>{walnut}</td>
      <td>juglans</td>
      <td>nut</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 65064,...</td>
      <td>protein</td>
      <td>39015.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>365</td>
      <td>366</td>
      <td>cucumber</td>
      <td>{cucumber}</td>
      <td>cucumis</td>
      <td>gourd</td>
      <td>{8194, 644104, 527, 8723, 42011, 31260, 536272...</td>
      <td>vegetable</td>
      <td>38952.0</td>
    </tr>
    <tr>
      <th>352</th>
      <td>353</td>
      <td>354</td>
      <td>cauliflower</td>
      <td>{cauliflower}</td>
      <td>brassica oleracea</td>
      <td>cabbage</td>
      <td>{644104, 527, 8723, 31260, 15394, 6184, 65064,...</td>
      <td>vegetable</td>
      <td>38890.0</td>
    </tr>
    <tr>
      <th>384</th>
      <td>393</td>
      <td>394</td>
      <td>pomegranate</td>
      <td>{pomegranate}</td>
      <td>punica</td>
      <td>fruit</td>
      <td>{644104, 527, 8723, 5372954, 31260, 15394, 528...</td>
      <td>fruit</td>
      <td>38886.0</td>
    </tr>
    <tr>
      <th>264</th>
      <td>264</td>
      <td>265</td>
      <td>sage</td>
      <td>{common sage, culinary sage, garden sage}</td>
      <td>salvia</td>
      <td>herb</td>
      <td>{644104, 5315594, 527, 8723, 31253, 522266, 31...</td>
      <td>vegetable</td>
      <td>38875.0</td>
    </tr>
    <tr>
      <th>365</th>
      <td>374</td>
      <td>375</td>
      <td>allspice</td>
      <td>{pimenta, myrtle pepper, allspice}</td>
      <td>pimenta</td>
      <td>spice</td>
      <td>{644104, 527, 8723, 31253, 522266, 31260, 8222...</td>
      <td>seasoning</td>
      <td>38863.0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>298</td>
      <td>299</td>
      <td>sesame</td>
      <td>{sesame}</td>
      <td>sesamum</td>
      <td>seed</td>
      <td>{5283329, 62465, 1030, 5283335, 1032, 644104, ...</td>
      <td>protein</td>
      <td>38850.0</td>
    </tr>
    <tr>
      <th>111</th>
      <td>111</td>
      <td>112</td>
      <td>hyssop oil</td>
      <td>{hyssop oil}</td>
      <td>hyssopus officinalis</td>
      <td>essential oil</td>
      <td>{5356544, 644104, 527, 8723, 31253, 522266, 31...</td>
      <td>seasoning</td>
      <td>38806.0</td>
    </tr>
    <tr>
      <th>371</th>
      <td>380</td>
      <td>381</td>
      <td>carom seed</td>
      <td>{ajwain, ajowan caraway, ajowan, bishops weed}</td>
      <td>ajwain</td>
      <td>spice</td>
      <td>{644104, 94217, 527, 4114, 8723, 31253, 31260,...</td>
      <td>seasoning</td>
      <td>38797.0</td>
    </tr>
    <tr>
      <th>349</th>
      <td>350</td>
      <td>351</td>
      <td>broccoli</td>
      <td>{broccoli}</td>
      <td>brassica oleracea</td>
      <td>cabbage</td>
      <td>{644104, 527, 8723, 5372954, 31260, 15394, 528...</td>
      <td>vegetable</td>
      <td>38792.0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>422</td>
      <td>426</td>
      <td>sunflower</td>
      <td>{helianthus}</td>
      <td>helianthus</td>
      <td>flower</td>
      <td>{644104, 5315594, 527, 8723, 5372954, 31260, 8...</td>
      <td>vegetable</td>
      <td>38769.0</td>
    </tr>
    <tr>
      <th>343</th>
      <td>344</td>
      <td>345</td>
      <td>leek</td>
      <td>{leek}</td>
      <td>allium ampeloprasum</td>
      <td>vegetable</td>
      <td>{1031, 644104, 31245, 527, 5320722, 8723, 1049...</td>
      <td>vegetable</td>
      <td>38730.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>50</td>
      <td>51</td>
      <td>barley</td>
      <td>{barley}</td>
      <td>hordeum</td>
      <td>cereal</td>
      <td>{644104, 77837, 527, 8723, 31252, 7704, 1049, ...</td>
      <td>grain</td>
      <td>38729.0</td>
    </tr>
    <tr>
      <th>225</th>
      <td>225</td>
      <td>226</td>
      <td>sour cherry</td>
      <td>{tart cherry, dwarf cherry, sour cherry}</td>
      <td>prunus</td>
      <td>berry</td>
      <td>{8193, 644104, 5281804, 8205, 527, 8723, 31253...</td>
      <td>fruit</td>
      <td>38725.0</td>
    </tr>
    <tr>
      <th>385</th>
      <td>394</td>
      <td>395</td>
      <td>poppy seed</td>
      <td>{poppy seed}</td>
      <td>papaver</td>
      <td>spice</td>
      <td>{644104, 527, 44229138, 8723, 31252, 31260, 53...</td>
      <td>seasoning</td>
      <td>38722.0</td>
    </tr>
    <tr>
      <th>259</th>
      <td>259</td>
      <td>260</td>
      <td>lemon balm</td>
      <td>{balm mint , common balm, balm}</td>
      <td>melissa officinalis</td>
      <td>herb</td>
      <td>{644104, 5315594, 527, 8723, 31253, 5144, 3126...</td>
      <td>vegetable</td>
      <td>38701.0</td>
    </tr>
    <tr>
      <th>279</th>
      <td>280</td>
      <td>281</td>
      <td>almond</td>
      <td>{almond}</td>
      <td>prunus</td>
      <td>nut</td>
      <td>{644104, 527, 8723, 5283349, 31260, 15394, 528...</td>
      <td>protein</td>
      <td>38665.0</td>
    </tr>
    <tr>
      <th>354</th>
      <td>355</td>
      <td>356</td>
      <td>mustard</td>
      <td>{mustard}</td>
      <td>brassica</td>
      <td>cabbage</td>
      <td>{644104, 527, 8723, 31260, 7710, 15394, 6184, ...</td>
      <td>vegetable</td>
      <td>38660.0</td>
    </tr>
    <tr>
      <th>386</th>
      <td>395</td>
      <td>396</td>
      <td>spinach</td>
      <td>{palak}</td>
      <td>spinacia</td>
      <td>vegetable</td>
      <td>{2762759, 644104, 527, 5283345, 8723, 5283349,...</td>
      <td>vegetable</td>
      <td>38597.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 9 columns</p>
</div>




```python
def food_color_map():
    return {
        'grain': "#ffe119", # yellow
        'vegetable': "#3cb44b", # green
        'fruit': "#f032e6", # purple
        'protein': "#e6194B", # red
        'seasoning': "#f58231", # orange
        'dairy': "#a9a9a9", # grey
        'fat': "#9A6324", # brown
        'sugar': "#800000", # maroon
        'beverage': "#4363d8", # blue
        'alcohol': "#42d4f4" # turquoise
    }
```


```python
ridf100_edges = flatten([
    [
        [ridf100.index[i], ridf100.index[j]]
        for j in range(1 + i, len(ridf100.index))
        if msc_matrix[ridf100.index[i]][ridf100.index[j]] > 0
    ]
    for i in range(len(ridf100.index))
])

edge_ridf100 = pd.DataFrame(ridf100_edges, columns=['a', 'b'])
edge_ridf100['weight'] = [
    msc_matrix[a][b]
    for a, b in zip(edge_ridf100['a'], edge_ridf100['b'])
]
edge_ridf100 = edge_ridf100.sort_values(by='weight', axis=0, ascending=False).head(100)

max_edge_weight = max(edge_ridf100['weight'])
edge_ridf100['weight'] = [
    w / max_edge_weight
    for w in edge_ridf100['weight']
]

# map each food group to a color gradient
fcmap = {
    k: get_color_gradient(Color('white'), Color(v), int(max_sum))
    for k, v in food_color_map().items()
}
```


```python
max_sum = max(ridf100['msc sum'])
sizes = [
    1000.0 * (msc_sum / max_sum)**10
    for msc_sum in ridf100['msc sum']
]

ncolors = [
    fcmap[g][int((mscs**2) / (max_sum**1)) - 1]
    for g, mscs in zip(ridf100['group'], ridf100['msc sum'])
]

widths = [
    2 * w**10
    for w in edge_ridf100['weight']
]
```


```python
make_network(
    edge_ridf100, 'a', 'b',
    nodelist=list(ridf100.index),
    node_sizes=sizes,
    node_colors=ncolors,
    node_tooltips=list(ridf100['alias']),
    linewidths=widths,
    title='Figure ' + str(fignum) + ':\nNetwork Graph of Food Groups'
)
fignum += 1
mpld3.display()

# Note: the axes here are meaningless.
# see https://github.com/mpld3/mpld3/issues/197
```






<style>

</style>

<div id="fig_el26461405864427224961826539233"></div>
<script>
function mpld3_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
   // already loaded: just create the figure
   !function(mpld3){
       
       mpld3.draw_figure("fig_el26461405864427224961826539233", {"width": 864.0, "height": 576.0, "axes": [{"bbox": [0.125, 0.125, 0.775, 0.755], "xlim": [-1.066094262429787, 1.101676781621089], "ylim": [-1.0969586458630327, 0.6876399441929745], "xdomain": [-1.066094262429787, 1.101676781621089], "ydomain": [-1.0969586458630327, 0.6876399441929745], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}, {"position": "left", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2646140586442720536", "lines": [], "paths": [], "markers": [], "texts": [{"text": "Figure 45:\nNetwork Graph of Food Groups", "position": [0.49999999999999994, 1.0137969094922736], "coordinates": "axes", "h_anchor": "middle", "v_baseline": "auto", "rotation": -0.0, "fontsize": 12.0, "color": "#000000", "alpha": 1, "zorder": 3, "id": "el2646140586304928120"}], "collections": [{"offsets": "data01", "xindex": 0, "yindex": 1, "paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "pathtransforms": [[26.10421277216434, 0.0, 0.0, 26.10421277216434, 0.0, 0.0], [20.209591237440307, 0.0, 0.0, 20.209591237440307, 0.0, 0.0], [31.622776601683793, 0.0, 0.0, 31.622776601683793, 0.0, 0.0], [23.642411769722692, 0.0, 0.0, 23.642411769722692, 0.0, 0.0], [22.980543165233183, 0.0, 0.0, 22.980543165233183, 0.0, 0.0], [21.396594619365928, 0.0, 0.0, 21.396594619365928, 0.0, 0.0], [22.692026163208965, 0.0, 0.0, 22.692026163208965, 0.0, 0.0], [21.974728458146124, 0.0, 0.0, 21.974728458146124, 0.0, 0.0], [20.425851519477668, 0.0, 0.0, 20.425851519477668, 0.0, 0.0], [18.523008506202807, 0.0, 0.0, 18.523008506202807, 0.0, 0.0], [17.118582018076925, 0.0, 0.0, 17.118582018076925, 0.0, 0.0], [17.102367786077792, 0.0, 0.0, 17.102367786077792, 0.0, 0.0], [16.02249511949902, 0.0, 0.0, 16.02249511949902, 0.0, 0.0], [19.078073769257752, 0.0, 0.0, 19.078073769257752, 0.0, 0.0], [20.137912973657944, 0.0, 0.0, 20.137912973657944, 0.0, 0.0], [20.311678421014488, 0.0, 0.0, 20.311678421014488, 0.0, 0.0], [18.503580590172056, 0.0, 0.0, 18.503580590172056, 0.0, 0.0], [18.744334289440204, 0.0, 0.0, 18.744334289440204, 0.0, 0.0], [13.873160280345468, 0.0, 0.0, 13.873160280345468, 0.0, 0.0], [19.329249683492105, 0.0, 0.0, 19.329249683492105, 0.0, 0.0], [14.467464541910172, 0.0, 0.0, 14.467464541910172, 0.0, 0.0], [18.91720288379755, 0.0, 0.0, 18.91720288379755, 0.0, 0.0], [19.268995067020498, 0.0, 0.0, 19.268995067020498, 0.0, 0.0], [18.31875688165127, 0.0, 0.0, 18.31875688165127, 0.0, 0.0], [16.68111825638485, 0.0, 0.0, 16.68111825638485, 0.0, 0.0], [18.735615183493184, 0.0, 0.0, 18.735615183493184, 0.0, 0.0], [14.658085828872162, 0.0, 0.0, 14.658085828872162, 0.0, 0.0], [14.939596005065855, 0.0, 0.0, 14.939596005065855, 0.0, 0.0], [17.44751657393009, 0.0, 0.0, 17.44751657393009, 0.0, 0.0], [14.856140865067832, 0.0, 0.0, 14.856140865067832, 0.0, 0.0], [14.088701062046443, 0.0, 0.0, 14.088701062046443, 0.0, 0.0], [13.868019915206538, 0.0, 0.0, 13.868019915206538, 0.0, 0.0], [16.94492194881305, 0.0, 0.0, 16.94492194881305, 0.0, 0.0], [15.85970516244388, 0.0, 0.0, 15.85970516244388, 0.0, 0.0], [15.789258313454033, 0.0, 0.0, 15.789258313454033, 0.0, 0.0], [15.01612132624782, 0.0, 0.0, 15.01612132624782, 0.0, 0.0], [13.777457749144302, 0.0, 0.0, 13.777457749144302, 0.0, 0.0], [15.903623026499575, 0.0, 0.0, 15.903623026499575, 0.0, 0.0], [15.985991296688118, 0.0, 0.0, 15.985991296688118, 0.0, 0.0], [12.960757081557189, 0.0, 0.0, 12.960757081557189, 0.0, 0.0], [13.451664506760286, 0.0, 0.0, 13.451664506760286, 0.0, 0.0]], "alphas": [null], "edgecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "facecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "edgewidths": [1.0], "offsetcoordinates": "data", "pathcoordinates": "display", "zorder": 1, "id": "el2646140586434720432"}, {"offsets": "data02", "xindex": 0, "yindex": 1, "paths": [[[[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.03166523806803117, 0.22287583372083936]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.318760996053751, 0.2819082979760076]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[-0.7054834085444432, 0.18954303097090672], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.02451477531011198, -0.5004370902208061]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.02451477531011198, -0.5004370902208061], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5567407308685969, -0.10161225237678856]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.49119981819387787, 0.4713556273889044]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7080929778253933, 0.10237591439134755]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.09672025164178313, 0.5882367558385014]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5960101020472115, 0.3458173903413681]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.694997321368688, -0.4518179988555059], [0.9880509268638601, -0.5646175440312478]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.5564968176194074, -0.41538504475016036]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.007296277833723948, 0.3317623934766252], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.9930971818144998, -0.3172040173972458]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]]], "pathtransforms": [], "alphas": [null], "edgecolors": ["#000000"], "facecolors": [], "edgewidths": [2.0, 2.0, 1.7971360049655776, 1.7028201750162486, 1.7028201750162486, 1.4459903108084542, 1.3684762094092209, 1.3684762094092209, 1.2947220788110518, 1.0944203670967418, 0.8217429221603281, 0.8217429221603281, 0.7310392864543159, 0.7310392864543159, 0.6891575988391958, 0.6891575988391958, 0.6494479051516725, 0.6118094974693024, 0.5761457776126219, 0.5761457776126219, 0.5423641125439713, 0.5423641125439713, 0.5423641125439713, 0.5103756940312796, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.4514416748028664, 0.4514416748028664, 0.4243363763225633, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.3515785515427856, 0.3515785515427856, 0.3515785515427856, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768], "offsetcoordinates": "display", "pathcoordinates": "data", "zorder": 2, "id": "el2646140586434721496"}], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344], [-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461], [-0.258852008859563, 0.22363802372680397], [-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845], [-0.27883126450534346, 0.4263192592828088], [-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672], [0.318760996053751, 0.2819082979760076], [-0.9555949862397308, 0.18309054095941033], [-0.025740726371598452, -0.17859720100749624], [-0.31667624392009824, 0.32298281058535117], [-0.12452945728936689, -0.031280906794874996], [0.13804208720886188, 0.42141163838398654], [0.007296277833723948, 0.3317623934766252], [-0.13091583957276873, -0.4941278178260166], [-0.3618467225483092, 0.44602602876073544], [-0.02451477531011198, -0.5004370902208061], [-0.5070249627126641, 0.2727432850826219], [-0.3073570292520302, -0.11803481585525133], [-0.007158935520356871, 0.07508200963637902], [0.22354677696600622, 0.40696833683734285], [0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134], [0.5564968176194074, -0.41538504475016036], [-0.31935710378641086, 0.06078581959112737], [0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059], [-0.5567407308685969, -0.10161225237678856], [-0.4784398294517299, 0.07381059019960214], [-0.49119981819387787, 0.4713556273889044], [0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0], [0.09672025164178313, 0.5882367558385014], [-0.5960101020472115, 0.3458173903413681], [0.9880509268638601, -0.5646175440312478], [0.9930971818144998, -0.3172040173972458]], "data02": [[0.0, 0.0]]}, "id": "el2646140586442722496", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}, {"type": "tooltip", "id": "el2646140586434720432", "labels": ["apple", "strawberry", "tea", "guava", "tomato", "cocoa", "grape", "mango", "potato", "green beans", "beans", "banana", "lima beans", "orange", "soybean", "papaya", "pineapple", "apricot", "mandarin orange", "mushroom", "lemon", "rice", "black currant", "passionfruit", "plum", "ginger", "pepper", "laurel", "peach", "rosemary", "basil", "oregano", "capsicum", "peanut", "raspberry", "spearmint", "peppermint", "melon", "corn", "marjoram", "nutmeg"], "hoffset": 0, "voffset": 10, "location": "mouse"}]});
   }(mpld3);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/mpld3
   require.config({paths: {d3: "https://mpld3.github.io/js/d3.v3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.3.1.dev1.js", function(){
         
         mpld3.draw_figure("fig_el26461405864427224961826539233", {"width": 864.0, "height": 576.0, "axes": [{"bbox": [0.125, 0.125, 0.775, 0.755], "xlim": [-1.066094262429787, 1.101676781621089], "ylim": [-1.0969586458630327, 0.6876399441929745], "xdomain": [-1.066094262429787, 1.101676781621089], "ydomain": [-1.0969586458630327, 0.6876399441929745], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}, {"position": "left", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2646140586442720536", "lines": [], "paths": [], "markers": [], "texts": [{"text": "Figure 45:\nNetwork Graph of Food Groups", "position": [0.49999999999999994, 1.0137969094922736], "coordinates": "axes", "h_anchor": "middle", "v_baseline": "auto", "rotation": -0.0, "fontsize": 12.0, "color": "#000000", "alpha": 1, "zorder": 3, "id": "el2646140586304928120"}], "collections": [{"offsets": "data01", "xindex": 0, "yindex": 1, "paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "pathtransforms": [[26.10421277216434, 0.0, 0.0, 26.10421277216434, 0.0, 0.0], [20.209591237440307, 0.0, 0.0, 20.209591237440307, 0.0, 0.0], [31.622776601683793, 0.0, 0.0, 31.622776601683793, 0.0, 0.0], [23.642411769722692, 0.0, 0.0, 23.642411769722692, 0.0, 0.0], [22.980543165233183, 0.0, 0.0, 22.980543165233183, 0.0, 0.0], [21.396594619365928, 0.0, 0.0, 21.396594619365928, 0.0, 0.0], [22.692026163208965, 0.0, 0.0, 22.692026163208965, 0.0, 0.0], [21.974728458146124, 0.0, 0.0, 21.974728458146124, 0.0, 0.0], [20.425851519477668, 0.0, 0.0, 20.425851519477668, 0.0, 0.0], [18.523008506202807, 0.0, 0.0, 18.523008506202807, 0.0, 0.0], [17.118582018076925, 0.0, 0.0, 17.118582018076925, 0.0, 0.0], [17.102367786077792, 0.0, 0.0, 17.102367786077792, 0.0, 0.0], [16.02249511949902, 0.0, 0.0, 16.02249511949902, 0.0, 0.0], [19.078073769257752, 0.0, 0.0, 19.078073769257752, 0.0, 0.0], [20.137912973657944, 0.0, 0.0, 20.137912973657944, 0.0, 0.0], [20.311678421014488, 0.0, 0.0, 20.311678421014488, 0.0, 0.0], [18.503580590172056, 0.0, 0.0, 18.503580590172056, 0.0, 0.0], [18.744334289440204, 0.0, 0.0, 18.744334289440204, 0.0, 0.0], [13.873160280345468, 0.0, 0.0, 13.873160280345468, 0.0, 0.0], [19.329249683492105, 0.0, 0.0, 19.329249683492105, 0.0, 0.0], [14.467464541910172, 0.0, 0.0, 14.467464541910172, 0.0, 0.0], [18.91720288379755, 0.0, 0.0, 18.91720288379755, 0.0, 0.0], [19.268995067020498, 0.0, 0.0, 19.268995067020498, 0.0, 0.0], [18.31875688165127, 0.0, 0.0, 18.31875688165127, 0.0, 0.0], [16.68111825638485, 0.0, 0.0, 16.68111825638485, 0.0, 0.0], [18.735615183493184, 0.0, 0.0, 18.735615183493184, 0.0, 0.0], [14.658085828872162, 0.0, 0.0, 14.658085828872162, 0.0, 0.0], [14.939596005065855, 0.0, 0.0, 14.939596005065855, 0.0, 0.0], [17.44751657393009, 0.0, 0.0, 17.44751657393009, 0.0, 0.0], [14.856140865067832, 0.0, 0.0, 14.856140865067832, 0.0, 0.0], [14.088701062046443, 0.0, 0.0, 14.088701062046443, 0.0, 0.0], [13.868019915206538, 0.0, 0.0, 13.868019915206538, 0.0, 0.0], [16.94492194881305, 0.0, 0.0, 16.94492194881305, 0.0, 0.0], [15.85970516244388, 0.0, 0.0, 15.85970516244388, 0.0, 0.0], [15.789258313454033, 0.0, 0.0, 15.789258313454033, 0.0, 0.0], [15.01612132624782, 0.0, 0.0, 15.01612132624782, 0.0, 0.0], [13.777457749144302, 0.0, 0.0, 13.777457749144302, 0.0, 0.0], [15.903623026499575, 0.0, 0.0, 15.903623026499575, 0.0, 0.0], [15.985991296688118, 0.0, 0.0, 15.985991296688118, 0.0, 0.0], [12.960757081557189, 0.0, 0.0, 12.960757081557189, 0.0, 0.0], [13.451664506760286, 0.0, 0.0, 13.451664506760286, 0.0, 0.0]], "alphas": [null], "edgecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "facecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "edgewidths": [1.0], "offsetcoordinates": "data", "pathcoordinates": "display", "zorder": 1, "id": "el2646140586434720432"}, {"offsets": "data02", "xindex": 0, "yindex": 1, "paths": [[[[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.03166523806803117, 0.22287583372083936]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.318760996053751, 0.2819082979760076]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[-0.7054834085444432, 0.18954303097090672], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.02451477531011198, -0.5004370902208061]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.02451477531011198, -0.5004370902208061], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5567407308685969, -0.10161225237678856]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.49119981819387787, 0.4713556273889044]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7080929778253933, 0.10237591439134755]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.09672025164178313, 0.5882367558385014]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5960101020472115, 0.3458173903413681]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.694997321368688, -0.4518179988555059], [0.9880509268638601, -0.5646175440312478]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.5564968176194074, -0.41538504475016036]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.007296277833723948, 0.3317623934766252], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.9930971818144998, -0.3172040173972458]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]]], "pathtransforms": [], "alphas": [null], "edgecolors": ["#000000"], "facecolors": [], "edgewidths": [2.0, 2.0, 1.7971360049655776, 1.7028201750162486, 1.7028201750162486, 1.4459903108084542, 1.3684762094092209, 1.3684762094092209, 1.2947220788110518, 1.0944203670967418, 0.8217429221603281, 0.8217429221603281, 0.7310392864543159, 0.7310392864543159, 0.6891575988391958, 0.6891575988391958, 0.6494479051516725, 0.6118094974693024, 0.5761457776126219, 0.5761457776126219, 0.5423641125439713, 0.5423641125439713, 0.5423641125439713, 0.5103756940312796, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.4514416748028664, 0.4514416748028664, 0.4243363763225633, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.3515785515427856, 0.3515785515427856, 0.3515785515427856, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768], "offsetcoordinates": "display", "pathcoordinates": "data", "zorder": 2, "id": "el2646140586434721496"}], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344], [-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461], [-0.258852008859563, 0.22363802372680397], [-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845], [-0.27883126450534346, 0.4263192592828088], [-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672], [0.318760996053751, 0.2819082979760076], [-0.9555949862397308, 0.18309054095941033], [-0.025740726371598452, -0.17859720100749624], [-0.31667624392009824, 0.32298281058535117], [-0.12452945728936689, -0.031280906794874996], [0.13804208720886188, 0.42141163838398654], [0.007296277833723948, 0.3317623934766252], [-0.13091583957276873, -0.4941278178260166], [-0.3618467225483092, 0.44602602876073544], [-0.02451477531011198, -0.5004370902208061], [-0.5070249627126641, 0.2727432850826219], [-0.3073570292520302, -0.11803481585525133], [-0.007158935520356871, 0.07508200963637902], [0.22354677696600622, 0.40696833683734285], [0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134], [0.5564968176194074, -0.41538504475016036], [-0.31935710378641086, 0.06078581959112737], [0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059], [-0.5567407308685969, -0.10161225237678856], [-0.4784398294517299, 0.07381059019960214], [-0.49119981819387787, 0.4713556273889044], [0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0], [0.09672025164178313, 0.5882367558385014], [-0.5960101020472115, 0.3458173903413681], [0.9880509268638601, -0.5646175440312478], [0.9930971818144998, -0.3172040173972458]], "data02": [[0.0, 0.0]]}, "id": "el2646140586442722496", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}, {"type": "tooltip", "id": "el2646140586434720432", "labels": ["apple", "strawberry", "tea", "guava", "tomato", "cocoa", "grape", "mango", "potato", "green beans", "beans", "banana", "lima beans", "orange", "soybean", "papaya", "pineapple", "apricot", "mandarin orange", "mushroom", "lemon", "rice", "black currant", "passionfruit", "plum", "ginger", "pepper", "laurel", "peach", "rosemary", "basil", "oregano", "capsicum", "peanut", "raspberry", "spearmint", "peppermint", "melon", "corn", "marjoram", "nutmeg"], "hoffset": 0, "voffset": 10, "location": "mouse"}]});
      });
    });
}else{
    // require.js not available: dynamically load d3 & mpld3
    mpld3_load_lib("https://mpld3.github.io/js/d3.v3.min.js", function(){
         mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.3.1.dev1.js", function(){
                 
                 mpld3.draw_figure("fig_el26461405864427224961826539233", {"width": 864.0, "height": 576.0, "axes": [{"bbox": [0.125, 0.125, 0.775, 0.755], "xlim": [-1.066094262429787, 1.101676781621089], "ylim": [-1.0969586458630327, 0.6876399441929745], "xdomain": [-1.066094262429787, 1.101676781621089], "ydomain": [-1.0969586458630327, 0.6876399441929745], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}, {"position": "left", "nticks": 11, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2646140586442720536", "lines": [], "paths": [], "markers": [], "texts": [{"text": "Figure 45:\nNetwork Graph of Food Groups", "position": [0.49999999999999994, 1.0137969094922736], "coordinates": "axes", "h_anchor": "middle", "v_baseline": "auto", "rotation": -0.0, "fontsize": 12.0, "color": "#000000", "alpha": 1, "zorder": 3, "id": "el2646140586304928120"}], "collections": [{"offsets": "data01", "xindex": 0, "yindex": 1, "paths": [[[[0.0, -0.5], [0.13260155, -0.5], [0.25978993539242673, -0.44731684579412084], [0.3535533905932738, -0.3535533905932738], [0.44731684579412084, -0.25978993539242673], [0.5, -0.13260155], [0.5, 0.0], [0.5, 0.13260155], [0.44731684579412084, 0.25978993539242673], [0.3535533905932738, 0.3535533905932738], [0.25978993539242673, 0.44731684579412084], [0.13260155, 0.5], [0.0, 0.5], [-0.13260155, 0.5], [-0.25978993539242673, 0.44731684579412084], [-0.3535533905932738, 0.3535533905932738], [-0.44731684579412084, 0.25978993539242673], [-0.5, 0.13260155], [-0.5, 0.0], [-0.5, -0.13260155], [-0.44731684579412084, -0.25978993539242673], [-0.3535533905932738, -0.3535533905932738], [-0.25978993539242673, -0.44731684579412084], [-0.13260155, -0.5], [0.0, -0.5]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]]], "pathtransforms": [[26.10421277216434, 0.0, 0.0, 26.10421277216434, 0.0, 0.0], [20.209591237440307, 0.0, 0.0, 20.209591237440307, 0.0, 0.0], [31.622776601683793, 0.0, 0.0, 31.622776601683793, 0.0, 0.0], [23.642411769722692, 0.0, 0.0, 23.642411769722692, 0.0, 0.0], [22.980543165233183, 0.0, 0.0, 22.980543165233183, 0.0, 0.0], [21.396594619365928, 0.0, 0.0, 21.396594619365928, 0.0, 0.0], [22.692026163208965, 0.0, 0.0, 22.692026163208965, 0.0, 0.0], [21.974728458146124, 0.0, 0.0, 21.974728458146124, 0.0, 0.0], [20.425851519477668, 0.0, 0.0, 20.425851519477668, 0.0, 0.0], [18.523008506202807, 0.0, 0.0, 18.523008506202807, 0.0, 0.0], [17.118582018076925, 0.0, 0.0, 17.118582018076925, 0.0, 0.0], [17.102367786077792, 0.0, 0.0, 17.102367786077792, 0.0, 0.0], [16.02249511949902, 0.0, 0.0, 16.02249511949902, 0.0, 0.0], [19.078073769257752, 0.0, 0.0, 19.078073769257752, 0.0, 0.0], [20.137912973657944, 0.0, 0.0, 20.137912973657944, 0.0, 0.0], [20.311678421014488, 0.0, 0.0, 20.311678421014488, 0.0, 0.0], [18.503580590172056, 0.0, 0.0, 18.503580590172056, 0.0, 0.0], [18.744334289440204, 0.0, 0.0, 18.744334289440204, 0.0, 0.0], [13.873160280345468, 0.0, 0.0, 13.873160280345468, 0.0, 0.0], [19.329249683492105, 0.0, 0.0, 19.329249683492105, 0.0, 0.0], [14.467464541910172, 0.0, 0.0, 14.467464541910172, 0.0, 0.0], [18.91720288379755, 0.0, 0.0, 18.91720288379755, 0.0, 0.0], [19.268995067020498, 0.0, 0.0, 19.268995067020498, 0.0, 0.0], [18.31875688165127, 0.0, 0.0, 18.31875688165127, 0.0, 0.0], [16.68111825638485, 0.0, 0.0, 16.68111825638485, 0.0, 0.0], [18.735615183493184, 0.0, 0.0, 18.735615183493184, 0.0, 0.0], [14.658085828872162, 0.0, 0.0, 14.658085828872162, 0.0, 0.0], [14.939596005065855, 0.0, 0.0, 14.939596005065855, 0.0, 0.0], [17.44751657393009, 0.0, 0.0, 17.44751657393009, 0.0, 0.0], [14.856140865067832, 0.0, 0.0, 14.856140865067832, 0.0, 0.0], [14.088701062046443, 0.0, 0.0, 14.088701062046443, 0.0, 0.0], [13.868019915206538, 0.0, 0.0, 13.868019915206538, 0.0, 0.0], [16.94492194881305, 0.0, 0.0, 16.94492194881305, 0.0, 0.0], [15.85970516244388, 0.0, 0.0, 15.85970516244388, 0.0, 0.0], [15.789258313454033, 0.0, 0.0, 15.789258313454033, 0.0, 0.0], [15.01612132624782, 0.0, 0.0, 15.01612132624782, 0.0, 0.0], [13.777457749144302, 0.0, 0.0, 13.777457749144302, 0.0, 0.0], [15.903623026499575, 0.0, 0.0, 15.903623026499575, 0.0, 0.0], [15.985991296688118, 0.0, 0.0, 15.985991296688118, 0.0, 0.0], [12.960757081557189, 0.0, 0.0, 12.960757081557189, 0.0, 0.0], [13.451664506760286, 0.0, 0.0, 13.451664506760286, 0.0, 0.0]], "alphas": [null], "edgecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "facecolors": ["#B647EA", "#7E60E5", "#3CB34A", "#9C51E8", "#61BC53", "#D146DC", "#9355E7", "#8D58E6", "#EFC34D", "#7FBE66", "#9B5CD9", "#6E74E3", "#9062D9", "#7565E4", "#C04DDB", "#7F60E5", "#7168E4", "#7367E4", "#7EA2E1", "#79BD62", "#7B9AE1", "#EDC155", "#7764E4", "#7069E3", "#707AE2", "#E79266", "#E49A7A", "#95C075", "#6D6FE3", "#95C076", "#9AC079", "#E49C7E", "#89BE6C", "#8E63D9", "#7588E2", "#95BF75", "#9CC07B", "#7486E2", "#8FBF71", "#E39F83", "#E39D80"], "edgewidths": [1.0], "offsetcoordinates": "data", "pathcoordinates": "display", "zorder": 1, "id": "el2646140586434720432"}, {"offsets": "data02", "xindex": 0, "yindex": 1, "paths": [[[[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.03166523806803117, 0.22287583372083936]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.318760996053751, 0.2819082979760076]], ["M", "L"]], [[[-0.7080929778253933, 0.10237591439134755], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[-0.7054834085444432, 0.18954303097090672], [-0.9555949862397308, 0.18309054095941033]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.13188813744882105, 0.07815023083248959]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [-0.02451477531011198, -0.5004370902208061]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.02451477531011198, -0.5004370902208061], [-0.13091583957276873, -0.4941278178260166]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.13804208720886188, 0.42141163838398654]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.20076592119999082, 0.3226695770172461]], ["M", "L"]], [[[0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.12452945728936689, -0.031280906794874996]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.27883126450534346, 0.4263192592828088]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5567407308685969, -0.10161225237678856]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.258852008859563, 0.22363802372680397], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.4784398294517299, 0.07381059019960214]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.49119981819387787, 0.4713556273889044]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.6316065019170678, -0.5270347967338358], [0.6839775063844419, -0.35930852332071134]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7080929778253933, 0.10237591439134755]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.10333368827707896, 0.19057471882911067], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [0.09672025164178313, 0.5882367558385014]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [-0.007158935520356871, 0.07508200963637902]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.5960101020472115, 0.3458173903413681]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31935710378641086, 0.06078581959112737]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [0.030793256258915207, 0.21121542547840344]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.10333368827707896, 0.19057471882911067]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.5070249627126641, 0.2727432850826219]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.694997321368688, -0.4518179988555059], [0.9880509268638601, -0.5646175440312478]], ["M", "L"]], [[[0.030793256258915207, 0.21121542547840344], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[-0.2631729930337319, 0.13260761758488857], [-0.7054834085444432, 0.18954303097090672]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.025740726371598452, -0.17859720100749624], [0.32967982090265163, -0.2841834835806305]], ["M", "L"]], [[[0.32967982090265163, -0.2841834835806305], [0.5564968176194074, -0.41538504475016036]], ["M", "L"]], [[[0.7739544082778421, -0.7738073906517415], [0.573292963857332, -0.5439092759095142]], ["M", "L"]], [[[0.007296277833723948, 0.3317623934766252], [0.22354677696600622, 0.40696833683734285]], ["M", "L"]], [[[-0.27883126450534346, 0.4263192592828088], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.3073570292520302, -0.11803481585525133]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [0.007296277833723948, 0.3317623934766252]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.6316065019170678, -0.5270347967338358]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[0.5564968176194074, -0.41538504475016036], [0.694997321368688, -0.4518179988555059]], ["M", "L"]], [[[-0.11235357040678001, 0.27940700241901845], [-0.258852008859563, 0.22363802372680397]], ["M", "L"]], [[[-0.31667624392009824, 0.32298281058535117], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.03166523806803117, 0.22287583372083936], [-0.31667624392009824, 0.32298281058535117]], ["M", "L"]], [[[0.6839775063844419, -0.35930852332071134], [0.9930971818144998, -0.3172040173972458]], ["M", "L"]], [[[-0.20076592119999082, 0.3226695770172461], [-0.3618467225483092, 0.44602602876073544]], ["M", "L"]], [[[-0.13188813744882105, 0.07815023083248959], [-0.025740726371598452, -0.17859720100749624]], ["M", "L"]]], "pathtransforms": [], "alphas": [null], "edgecolors": ["#000000"], "facecolors": [], "edgewidths": [2.0, 2.0, 1.7971360049655776, 1.7028201750162486, 1.7028201750162486, 1.4459903108084542, 1.3684762094092209, 1.3684762094092209, 1.2947220788110518, 1.0944203670967418, 0.8217429221603281, 0.8217429221603281, 0.7310392864543159, 0.7310392864543159, 0.6891575988391958, 0.6891575988391958, 0.6494479051516725, 0.6118094974693024, 0.5761457776126219, 0.5761457776126219, 0.5423641125439713, 0.5423641125439713, 0.5423641125439713, 0.5103756940312796, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.48009540247558397, 0.4514416748028664, 0.4514416748028664, 0.4243363763225633, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.39870467645689395, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.37447492824689804, 0.3515785515427856, 0.3515785515427856, 0.3515785515427856, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.32994991978792426, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.30952625030746905, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.29024749801429434, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.27205625244654014, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.2548976380527078, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.23871921764182866, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.22347089891782493, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.20910484401873153, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768, 0.19557538198299768], "offsetcoordinates": "display", "pathcoordinates": "data", "zorder": 2, "id": "el2646140586434721496"}], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[-0.03166523806803117, 0.22287583372083936], [0.030793256258915207, 0.21121542547840344], [-0.2631729930337319, 0.13260761758488857], [-0.13188813744882105, 0.07815023083248959], [-0.20076592119999082, 0.3226695770172461], [-0.258852008859563, 0.22363802372680397], [-0.10333368827707896, 0.19057471882911067], [-0.11235357040678001, 0.27940700241901845], [-0.27883126450534346, 0.4263192592828088], [-0.7080929778253933, 0.10237591439134755], [-0.7054834085444432, 0.18954303097090672], [0.318760996053751, 0.2819082979760076], [-0.9555949862397308, 0.18309054095941033], [-0.025740726371598452, -0.17859720100749624], [-0.31667624392009824, 0.32298281058535117], [-0.12452945728936689, -0.031280906794874996], [0.13804208720886188, 0.42141163838398654], [0.007296277833723948, 0.3317623934766252], [-0.13091583957276873, -0.4941278178260166], [-0.3618467225483092, 0.44602602876073544], [-0.02451477531011198, -0.5004370902208061], [-0.5070249627126641, 0.2727432850826219], [-0.3073570292520302, -0.11803481585525133], [-0.007158935520356871, 0.07508200963637902], [0.22354677696600622, 0.40696833683734285], [0.32967982090265163, -0.2841834835806305], [0.6839775063844419, -0.35930852332071134], [0.5564968176194074, -0.41538504475016036], [-0.31935710378641086, 0.06078581959112737], [0.6316065019170678, -0.5270347967338358], [0.573292963857332, -0.5439092759095142], [0.694997321368688, -0.4518179988555059], [-0.5567407308685969, -0.10161225237678856], [-0.4784398294517299, 0.07381059019960214], [-0.49119981819387787, 0.4713556273889044], [0.7739544082778421, -0.7738073906517415], [0.9572333762852098, -1.0], [0.09672025164178313, 0.5882367558385014], [-0.5960101020472115, 0.3458173903413681], [0.9880509268638601, -0.5646175440312478], [0.9930971818144998, -0.3172040173972458]], "data02": [[0.0, 0.0]]}, "id": "el2646140586442722496", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}, {"type": "tooltip", "id": "el2646140586434720432", "labels": ["apple", "strawberry", "tea", "guava", "tomato", "cocoa", "grape", "mango", "potato", "green beans", "beans", "banana", "lima beans", "orange", "soybean", "papaya", "pineapple", "apricot", "mandarin orange", "mushroom", "lemon", "rice", "black currant", "passionfruit", "plum", "ginger", "pepper", "laurel", "peach", "rosemary", "basil", "oregano", "capsicum", "peanut", "raspberry", "spearmint", "peppermint", "melon", "corn", "marjoram", "nutmeg"], "hoffset": 0, "voffset": 10, "location": "mouse"}]});
            })
         });
}
</script>




```python

```




    (0.0, 1.0, 0.0, 1.0)




![png](food-tutorial_files/food-tutorial_50_1.png)



```python
# use colour & networkx to make a graph with varying sizes of nodes & colors to map flavors
# import csv file of recipes
# algorithm to predict flavors of cuisines - generate new recipes, fusion cuisine
# bonus - predict ratings on recipes
```
