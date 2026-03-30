Dataset description:

    Dataset is about scientific papers
    • Source DBLP (https://dblp.org/)
    • Contains over 6.7 M of papers




Key Schema Fields:
The dataset includes various metadata fields for each paper:


    id: Paper ID (string) 
    title: Paper title (string) 
    authors.name & author.id: Author names and their unique IDs 
    author.org: Author affiliation 
    venue & year: Publication venue name and year published 
    references: List of paper references 
    n_citation: Total citation number 
    doc_type: Paper type, such as journal or conference 
    abstract: Abstract text


Project Objectives:

    To successfully build the classification model, the following milestones must be met:
    Data Exploration and Visualization: Analyze the dataset to understand distributions and improve overall data quality.
    Feature Extraction: Derive meaningful variables from the raw JSONL text, metadata, and citation network.
    Feature Selection: Identify the most predictive features for the classification task.
    Classification: Build and train a machine learning model to predict citation links between pairs of papers


Known Data Quality Challenges

    During the data exploration phase, special attention must be paid to several known inconsistencies within the DBLP dataset:
    Missing Values: Frequent missing data, particularly missing author IDs and missing organization (ORG) information.
    Entity Resolution: Author names can be spelled differently across entries (e.g., "Mitrovic" vs. "Mitrović").
    Temporal Inconsistencies: An author's affiliated organization can change over time, or be present in some years and missing in others.Categorization Errors: Incorrect document types, such as a paper from an international conference being incorrectly labeled as a "Journal" in the doc_type field.


Follow up objectives:

WOrk on the feauture stream
Work on model stream
Work from simple to more complicated
Final network on citations
Then use the final network for featurs itself 