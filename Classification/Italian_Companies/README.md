# Overview 

This project is about cleaning a sample of Italian company records

Each record contains the information company such as: 

1. Business name 
2. Address 
3. City 
4. Province 
5. Region 
6. Country 

The main goal is to detect incorrect or missing location information
and produce a cleaned version of the records.

**The important point of this task is not only to fix the 500 sample records, but also to design 
an approach that could work on a much larger dataset, for example 4+ million records.**

---

## Files: 

The input data was provided in three formats:

- `db_data.json`
- `db_data.csv`
- `db_data.xlsx`

I used the JSON file as the main input, because the expected output is also close to JSON format.  
However, the same logic can also be applied to the CSV or Excel files.

The script produces:

- `cleaned_companies.json`
- `cleaned_companies.csv`
- `uncertain_records.csv`
- `uncertain_records.json`

---

## Initial Data Analysis

The dataset contains:

- 500 rows
- 6 columns

The columns are: 

['ragione_sociale', 'address', 'city', 'province', 'region', 'nation']


After checking missing values, I found that the main problems are related to location fields:

some records have missing `city`
some records have missing `province`
some records have missing `region`
some records have missing `address`
some provinces are `written as short codes`, for example BA, MI, RM
some regions `have inconsistent formatting`, for example different capitalization

The company `name` and `nation` fields `were mostly already complete`.


---

# Cleaning Approach

My approach is based on deterministic rules: 

- Most addresses follow this structure:

    - **STREET - CAP - CITY (PROVINCE_CODE)**     EX:  `VIA SOAVE 7 - 20135 - MILANO (MI)`


- From this address, I can extract:

    - CAP = 20135
    - city = MILANO
    - province code = MI


- Then I use a local mapping to convert:

    - MI -> Milano -> Lombardia
    - BA -> Bari -> Puglia
    - RM -> Roma -> Lazio


- So the cleaning process is:

  1. Load the dataset
  2. Check missing values
  3. Extract CAP, city, and province code from the address using regex
  4. Convert province codes into full province names
  5. Infer the region from the province
  6. Create cleaned columns for city, province, and region
  7. Save the final cleaned dataset
  8. Save uncertain records separately

---

## The script can fix cases like this:

### Before:
```
{
  "ragione_sociale": "CARUSO MAURO",
  "address": "PIAZZA DELLA RESISTENZA 9/10 -  - CONVERSANO (BA)",
  "city": null,
  "province": null,
  "region": null,
  "nation": "Italia"
}
```

### After: 
```
{
  "ragione_sociale": "CARUSO MAURO",
  "address": "PIAZZA DELLA RESISTENZA 9/10 -  - CONVERSANO (BA)",
  "city": "Conversano",
  "province": "Bari",
  "region": "Puglia",
  "nation": "Italia"
}
```

## The script also standardizes province values like:

- Ba -> Bari
- MI -> Milano
- Rm -> Roma

And it fills missing regions when the province is known.

---

### Assumptions

I made these assumptions:

1. If the city is missing but it exists in the address, I use the city extracted from the address.
2. If the province is written as a two-letter code, I convert it to the full province name.
3. If the province is known, I infer the region from the province.
4. If there is not enough information, I do not guess the value.
5. Records with missing address and missing location fields are marked as uncertain.
6. Foreign or unusual province codes such as EE are not forced into an Italian region.

---

### Uncertain Records

Some records cannot be fixed safely.

For example, if a record has no address, no city, no province, and no region, the script cannot know the correct location.

Instead of guessing, I export these records to:

- `uncertain_records.csv`
- `uncertain_records.json`


These records would need manual review or an external data source.

---

### Scalability

This solution is designed to be scalable.

I avoided using an external API call for every record because the full database contains more than 4 million companies.
Calling an external geocoding API for every row would be slow, expensive, and possibly limited by rate limits.

Instead, I used:

- regex parsing
- local dictionaries
- pandas operations
- local province and region mappings

This means the script can process many records without depending on external services.

For a production version, I would replace the manually written province and region dictionaries with an **official static dataset**, for example from ISTAT or another reliable Italian administrative source.

The static reference dataset would be loaded once and then used for all records.

#### Point: 
If we tried to use APIs, we would face with following problems:

1. Slowness due to `limited requests in a given time` & `Network delays waitng for responses from the API`
2. Cost in the case of using paid APIs
3. Reliability can be compromised in the case that the API service is down or unavailable. 
4. It will not be scalable in the case of dealing with large datasets as all the request need to be processed one by one


--- 

### Limitations

The current solution has some limitations:

- It depends on the address having a recognizable structure.
- It cannot fix records where the address is completely missing.
- It may not correctly handle all foreign addresses.
- It does not verify whether the street itself is correct.
- It does not use fuzzy matching for misspelled city names yet.

A possible improvement would be to use an official municipalities dataset containing:

- city name
- CAP
- province code
- province name
- region

This would make validation more accurate.

Another possible improvement would be to use external geocoding only for uncertain records, not for the whole dataset.


