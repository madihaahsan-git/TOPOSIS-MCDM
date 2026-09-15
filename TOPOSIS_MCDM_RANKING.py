import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# take  30 drugs (alternatives) and 12 properties (criteria)
drugs = ["Afatinib","Bleomycin","Capecitabine","Cyclophosphamide","Cabozantinib","Docetaxel","Doxorubicin","Erlotinib","Epirubicin",
         "Etoposide",
"5-flulorouraci","Gefitinib","Gemcitabine","Hydroxyurea","Irinotecan","(+)-Ifosfamide","folinic acid","Methotrexate","Mitomycin C",
              "Mitoxantrone","Palbociclib","Pemetrexed","Paclitaxel","Raltitrexed","Sorafenib","Taxanes","Temsirolimus","Vincristine",
              "(+)-Vinblastine","Vinorelbine"]
criteria = ["Index 1","Index 2","Index 3","Index 4","Index 5","Index 6","Index 7","Index 8","Index 9","Index 10","Index 11", "Index 12"]

n = len(criteria)
data = {
"NI": [180.60369,759.00792,123.31738,45.51713,230.32051,331.80805,198.09917,144.80010,198.09917,234.34716,24.50667,170.88514,69.14744,8.70820,276.36865,
     47.43467,193.27525,187.11815,97.75036,164.89418,184.23459,172.15501,368.37794,176.59187,179.04614,253.23886,467.05276,335.54434,330.54434,319.86767],

"IN1": [7.63657,13.54616,5.53045,4.33567,7.35842,12.04379,9.40116,6.69260,9.40116,9.90025,3.31829,6.82161,5.24885,1.84164,8.76384,4.16518,6.76238,
     6.60559,7.50212,7.06424,7.48725,6.38470,12.62727,6.60413,6.52305,12.02477,12.45239,13.86077,13.66077,13.28609],

"IN2": [90.22031,379.44326,61.60001,22.66329,115.10581,165.79154,98.93462,72.33066,98.93462,117.07029,12.16692,85.37821,34.48449,4.28634,138.12068,
     23.62273,96.58193,93.50416,48.75900,82.37999,92.04027,86.01746,184.08080,88.23431,89.46102,126.48895,233.45237,167.64168,165.14368,159.80695],

"MRI": [5.26754,9.29524,3.60120,2.81907,4.77322,8.51777,6.64967,4.22189,6.64967,6.71274,1.81188,4.49977,3.30137,0.95351,5.54156,2.94836,4.46621,
     4.35535,4.27872,4.37346,5.16637,4.38642,8.77208,4.51994,4.61336,7.67873,7.80463,8.77320,8.63175,8.50042],
"ND":  [138712,5024232,83672,4532,358898,367782,113148,90628,113148,179202,962,
      154148,11886,106,443918,5904,240442,229948,16524,124476,155580,187182,
      464840,187594,206274,145256,1131106,265198,261298,249720],

"KA":  [443.61173,2872.55537,294.72768,74.19254,651.69801,878.92006,458.98200,340.67932,458.98200,575.50663,33.26478,432.68121,126.17791,9.34847,785.64826,
     80.68704,522.70452,504.21033,177.58170,402.67895,461.62638,452.69637,1002.92246,462.18762,475.68361,586.40658,1444.54943,831.86556,819.37556,790.88419],

"MPI": [255.35453,1073.35637,174.35562,64.30353,325.68389,469.16789,280.07323,204.72919,280.07323,331.34389,34.59649,241.62257,97.72612,12.26722,390.79929,
     67.01582,273.29309,264.58619,138.15770,233.14813,260.49259,243.42148,520.88859,249.69504,253.16559,358.04156,660.46002,474.43907,
     467.36942,452.27147],

"DR": [127.64843,536.65672,87.15705,32.11802,162.82269,234.54416,139.99595,102.34005,139.99595,165.63541,17.26758,120.78852,48.83148,6.10947,
      195.37714,33.47441,136.62685,132.27368,69.03773,116.55033,130.21905,121.68950,260.40604,124.82573,126.56085,178.97463,330.20384,
      237.17339,233.63927,226.09086],

"HN":  [3.17601,3.66415,2.37374,2.70377,2.66374,4.63776,4.14134,2.91401,4.14134,4.11280,2.45635,2.76036,2.91568,1.70000,3.16300,2.49821,2.56307,2.51654,
      4.19367,2.95721,3.05423,2.49474,4.71883,2.58896,2.53009,5.29175,4.10868,5.68465,5.60465,5.46426],

"N4I": [3.17917,3.66481,2.37620,2.71554,2.66508,4.64120,4.14661,2.91695,4.14661,4.11676,2.47392,2.76257,2.92350,1.72474,3.16458,2.50920,2.56463,2.51811,
      4.20396,2.95976,3.05707,2.49672,4.72179,2.59096,2.53210,5.29750,4.11003,5.68936,5.60930,5.46886],

"SBN": [888,5746,590,149,1304,1759,919,682,919,1152,67,866,253,19,1572,162,1046,1009,356,806,924,906,2007,925,952,1174,2890,1665,1640,1583],

"PBN": [5472,83962,3456,404,10671,12570,5032,3868,5032,7088,125,5687,856,22,13012,480,7822,7504,1196,4924,5919,6417,15116,6489,6898,6430,28211,10397,10241,9835],


}
df = pd.DataFrame(data,index = drugs)
print("\nStep 1 (Nirmla index table):\n")
pd.set_option('display.expand_frame_repr', False)
print(df)
data_values = df.values
columns_sum = data_values.sum(axis=0)
Norm_entropy = data_values / columns_sum
norm_entropy = pd.DataFrame(Norm_entropy, columns=df.columns, index=df.index)
print("\nStep 2 (NORMALIZATION (for entropy)):\n")
Norm_entropy_df = pd.DataFrame(Norm_entropy, columns=df.columns, index=df.index)
pd.set_option('display.expand_frame_repr', False)
print(Norm_entropy_df)
m, n = norm_entropy.shape  
P = norm_entropy / norm_entropy.sum(axis=0)
k = 1 / np.log(m)
entropy = -k * (P * np.log(P)).sum(axis=0)
d = 1 - entropy
weights = d / d.sum()
results_table = pd.DataFrame({
    "Entropy": entropy,
    "Diversification (d)": d,
    "Weight": weights
})

print("\n=== Step 5 :  Entropy Method Results Table ===")
pd.set_option('display.expand_frame_repr', False)
print(results_table)
norm_topsis = df / np.sqrt((df**2).sum())

print("\n=== Step 6: Normalized Matrix (TOPSIS) ===")
pd.set_option('display.expand_frame_repr', False)
print(norm_topsis)
weighted = norm_topsis * weights

print("\n=== Step 7: Weighted Matrix ===")
pd.set_option('display.expand_frame_repr', False)
print(weighted)
ideal_best = weighted.min(axis=0).values
ideal_worst = weighted.max(axis=0).values
ideal_df = pd.DataFrame({
    "Criteria": weighted.columns,
    " I+ ": ideal_best,
    " I-": ideal_worst
})
print("\nStep 8:\n")
pd.set_option('display.expand_frame_repr', False)
print(ideal_df)

dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

score = dist_worst / (dist_best + dist_worst)
results = pd.DataFrame({
    "D+": dist_best,
    "D-": dist_worst,
    "Score": score
})

results["Rank"] = results["Score"].rank(ascending=False)
results = results.sort_values(by="Rank")
pd.set_option('display.expand_frame_repr', False)

print(results)
criteria = weights.index
values = weights.values
drug_names = ["Hydroxyurea","5-flulorouracil","Cyclophosphamide","(+)-Ifosfamide","Gemcitabine","Mitomycin","Capecitabine","Erlotinib","Mitoxantrone",
        "Gefitinib","Afatinib","Epirubicin","Doxorubicin","Palbociclib","Pemetrexed","Raltitrexed","Sorafenib","Methotrexate","folinic acid",
        "Etoposide","Taxanes","Cabozantinib","Vinorelbine","(+)-Vinblastine","Vincristine","Irinotecan",
        "Docetaxel","Paclitaxel","Temsirolimus","Bleomycin"]
scores = results["Score"]

colors = plt.cm.plasma(np.linspace(0, 1, len(scores)))

plt.figure(figsize=(14,6))
plt.bar(drug_names,scores.values, color=colors)

plt.xticks(rotation=90)
plt.xlabel("Alternatives")
plt.ylabel("TOPSIS Score")
plt.title("MCDM Score Comparison")

plt.tight_layout()
plt.savefig("drug_scores_comparison.png", dpi =300)
plt.show()
