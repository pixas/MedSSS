medqa_prompt = [{
    "problem": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\nA. Disclose the error to the patient but leave it out of the operative report\nB. Disclose the error to the patient and put it in the operative report\nC. Tell the attending that he cannot fail to disclose this mistake\nD. Report the physician to the ethics committee\nE. Refuse to dictate the operative report",
    "answer": """The correct next action for the resident to take is:

**B. Disclose the error to the patient and put it in the operative report.**

Here's the reasoning:

- **Disclosing errors** is a crucial part of medical ethics and patient care. When a mistake is made, especially one that involves a risk, the patient has a right to be informed, even if the error does not have any lasting consequences. Failing to disclose can undermine trust between the patient and the healthcare provider, and it goes against the principles of honesty and transparency.
  
- **Documenting errors in the operative report** is also important. This helps maintain an accurate medical record for continuity of care and legal protection, and it is a professional and ethical obligation to ensure that all significant events during surgery are recorded.
  
The resident should act in accordance with ethical guidelines, which prioritize transparency and accountability. While the attending's directive is to omit the error from the report, the resident's responsibility is to ensure the accuracy of the medical record and to maintain patient trust through full disclosure. The answer is B."""
}]

pubmedqa_prompt = [{
    "problem": "Question: Is anorectal endosonography valuable in dyschesia?\n\nA. yes\nB. no\nC. maybe",
    "answer": """To address this question, let's break it down step by step:

1. **Understanding Anorectal Endosonography**: This is a diagnostic procedure that uses high-frequency ultrasound to visualize the anorectal area, including the anal sphincters, rectal wall, and surrounding structures. It's commonly used in the assessment of conditions such as fecal incontinence, anorectal cancer, and other rectal diseases.

2. **Understanding Dyschesia**: Dyschesia refers to painful or difficult defecation, which may arise from various causes, including constipation, anorectal dysfunction, or pelvic floor disorders.

3. **Role of Anorectal Endosonography in Dyschesia**:
   - Anorectal endosonography can help in identifying structural abnormalities such as issues with the anal sphincter, rectal wall, or other anatomical abnormalities that could contribute to difficulty in passing stool.
   - It is particularly useful for evaluating conditions like anorectal malformations or sphincter abnormalities that could be causing dysfunction in the defecation process.

4. **Clinical Relevance**: In cases of dyschesia where there are underlying structural issues with the anal sphincter or pelvic floor, anorectal endosonography can provide valuable information to guide treatment, including biofeedback therapy or surgical intervention.

### Conclusion:
Based on the diagnostic value of anorectal endosonography in identifying structural causes of dyschesia, it is indeed valuable in assessing patients with this condition.

### Final Answer:
**A. yes**"""
}]

medmcqa_prompt = [{
    "problem": "Which of the following is not true for myelinated nerve fibers:\nA. Impulse through myelinated fibers is slower than non-myelinated fibers\nB. Membrane currents are generated at nodes of Ranvier\nC. Saltatory conduction of impulses is seen\nD. Local anesthesia is effective only when the nerve is not covered by myelin sheath",
    "answer": """Let's break down each statement to identify which one is not true for myelinated nerve fibers:

1. **A. Impulse through myelinated fibers is slower than non-myelinated fibers**  
   This is **not true**. Myelinated fibers transmit impulses faster than non-myelinated fibers. The myelin sheath acts as an insulating layer, allowing the electrical signal to travel faster along the fiber by jumping between the nodes of Ranvier (saltatory conduction).

2. **B. Membrane currents are generated at nodes of Ranvier**  
   This is **true**. In myelinated fibers, action potentials are only generated at the nodes of Ranvier, where the axonal membrane is exposed. These nodes are essential for saltatory conduction, where the action potential "jumps" from one node to the next.

3. **C. Saltatory conduction of impulses is seen**  
   This is **true**. Saltatory conduction is the rapid transmission of action potentials in myelinated nerve fibers, where the impulse jumps from one node of Ranvier to the next, speeding up the signal.

4. **D. Local anesthesia is effective only when the nerve is not covered by myelin sheath**  
   This is **true**. Local anesthetics work by blocking the propagation of action potentials in nerve fibers. In myelinated fibers, the anesthetic works effectively at the nodes of Ranvier, where the action potentials are generated. However, the anesthetic might not work as efficiently on myelinated segments compared to non-myelinated fibers.

### Final Answer:
The statement that is **not true** is:  
**A. Impulse through myelinated fibers is slower than non-myelinated fibers.**"""
}]


pubhealth_prompt = [{
    "problem": "Infer whether text1 is a fact that can be inferred from text2.\nText1: \"\"\"The money the Clinton Foundation took from from foreign governments while Hillary Clinton was secretary of state \"\"\"\"is clearly illegal. \u2026 The Constitution says you can\u2019t take this stuff.\"\"\"\nText2: \"\"\"Gingrich said the Clinton Foundation \"\"\"\"took money from from foreign governments while (Hillary Clinton) was secretary of state. It is clearly illegal. \u2026 The Constitution says you can\u2019t take this stuff.\"\"\"\" A clause in the Constitution does prohibit U.S. officials such as former Secretary of State Hillary Clinton from receiving gifts, or emoluments, from foreign governments. But the gifts in this case were donations from foreign governments that went to the Clinton Foundation, not Hillary Clinton. She was not part of the foundation her husband founded while she was secretary of state. Does that violate the Constitution? Some libertarian-minded constitutional law experts say it very well could. Others are skeptical. What\u2019s clear is there is room for ambiguity, and the donations are anything but \"\"\"\"clearly illegal.\"\"\"\" The reality is this a hazy part of U.S. constitutional\u00a0law. \"\nA. true, text1 is a fact.\nB. false, text1 is not a fact.\nC. mixture, text1 contains both fact and non-fact information.",
	"answer": """Let's break down the two texts:

### Text1:
- **Claim**: "The money the Clinton Foundation took from foreign governments while Hillary Clinton was secretary of state is clearly illegal... The Constitution says you can’t take this stuff."
- This is a statement asserting that donations to the Clinton Foundation from foreign governments are "clearly illegal," citing the Constitution as evidence.

### Text2:
- **Claim**: "Gingrich said the Clinton Foundation took money from foreign governments while (Hillary Clinton) was secretary of state. It is clearly illegal... The Constitution says you can’t take this stuff."
- **Analysis**: Text2 also mentions the Clinton Foundation receiving donations from foreign governments while Hillary Clinton was Secretary of State and highlights the Constitutional prohibition on foreign gifts to U.S. officials. However, Text2 elaborates that the donations went to the Clinton Foundation, not directly to Hillary Clinton herself, and there is ambiguity around whether this constitutes a violation of the Constitution.
- **Ambiguity**: Text2 explicitly states there is room for ambiguity and suggests that the donations might not be "clearly illegal" as Text1 claims.

### Conclusion:
Text2 does not support the "clearly illegal" assertion made in Text1. Instead, it introduces doubt and mentions differing legal opinions, suggesting that the situation is not as straightforward as Text1 implies.

Therefore, the answer is:
**B. false, text1 is not a fact.**"""
 }]

healthfact_prompt = [{
    "problem": "Given a paragraph and a claim, classify it this way: If the claim contradicts the evidence present in the paragraph, classify the claim as '0'. If the claim has multiple supporting *AND* contradicting evidences, classify the claim as '1'. If the claim has supporting evidence and the paragraph is in overall favor of the claim, then classify the claim as '2'.\nParagraph: While the financial costs of screening mammography & false-positive recalls & biopsies were not included in the study, readers would benefit from being reminded that recall mammography & biopsies increase patient financial costs. This article leads with valuable information for all women over age 40 by stating \u201cGetting screening mammograms every two years instead of annually reduces the chance of a false alarm, a new study shows.\u201d\u00a0 Unfortunately the writer doesn\u2019t quantify or elaborate adequately on that reduction. Instead, the writer later focuses on how women undergoing screening mammography shouldn\u2019t be anxious if they experience a recall because over half of women who have screening mammograms for 10 years will be recalled. Readers would have been better served if the writer had emphasized the significant reduction in both recall mammograms & false-positive biopsies in women who are screened every two years rather than annually. Part of the continuing controversy\u00a0over screening mammography\u00a0focuses on\u00a0annual versus biennial screening. Because this\u00a0study showed a significant reduction in the percentage of those recalled & needing biopsies among women who are screened\u00a0every other year, with no\u00a0statistically significant increase in late-stage diagnosis of breast cancer, the article should\u00a0have emphasized those important findings. The piece states that the researchers noted that \u201c\u2026false positive recalls may cause inconvenience & anxiety and biopsies can cause pain and scarring.\u201d This article fails to include several important facts about the study. 1) This is a prospective cohort study of women screened between 1994-2006. 2) Most of the mammograms were film rather than digital. 3) Few women underwent screening for the entire 10 year period. 4) Screening mammography recall rates are influenced by the skill of the radiologists who read the mammograms. There was no disease mongering. No independent experts were quoted. The story did an adequate job comparing \u2013 at a very high level \u2013 annual screening versus biennial screening. Screening mammography is widely available throughout the United States. The study did not examine a new procedure. There\u2019s no evidence that the story relied solely on a news release. Claim: Annual Mammograms May Have More False-Positives",
	"answer": """Let's analyze the paragraph and the claim step by step:

### Step 1: Identify the key points in the paragraph
- The article discusses the pros and cons of screening mammography, specifically comparing annual versus biennial screening.
- It mentions that biennial (every two years) screening leads to fewer false-positive recalls and biopsies, which is considered an important finding that the article fails to emphasize adequately.
- The paragraph highlights the financial cost of mammography, false-positive recalls, and biopsies but does not explicitly calculate or elaborate on the reduction of false positives.
- It also points out that the article fails to mention important details about the study, such as the study's duration and the types of mammograms used.

### Step 2: Examine the claim
**Claim:** Annual Mammograms May Have More False-Positives

This claim is suggesting that annual mammograms may lead to more false-positive results. 

### Step 3: Look for evidence supporting or contradicting the claim
- **Supporting evidence:** The paragraph mentions that "biennial screening" (every two years) reduces the chance of false-positive recalls and biopsies. This suggests that annual screening may result in more false positives, supporting the claim.
- **Contradicting evidence:** The paragraph does not explicitly state that annual screening *definitely* leads to more false positives. It emphasizes the reduction in false positives when switching to biennial screening, but it doesn't provide direct evidence that annual screening is worse than biennial in terms of false positives.

### Step 4: Classify the claim
- The claim has **supporting evidence**: The paragraph suggests that biennial screening leads to fewer false positives, which indirectly supports the idea that annual screening might have more false positives.
- The claim does not have **contradicting evidence**: The paragraph does not explicitly dispute the claim but rather points out an underemphasized aspect of the study (the reduction of false positives with biennial screening).

### Conclusion:
Since the claim is supported by the paragraph (indirectly through the comparison with biennial screening), and there is no direct contradiction, the classification is **2** (The claim is supported by the evidence and the paragraph is overall in favor of the claim).

### Final Answer: **2**"""
 }]


biomrc_prompt = [{
    "problem": "Below is a passage where medical entities are replaced by placeholder. You need to read the passage carefully and determine the most suitable title for this passage.\nAbstract: Single-agent activity for @entity12 reflected by response rates of 10%-30% has been reported in @entity0 with @entity3 ( @entity3 ). Preclinical data indicate that alpha-interferon could enhance the cytotoxic activity of the @entity11 @entity6 or its analog @entity7 . In a phase I/II study, 31 @entity0 with biopsy-proven inoperable @entity3 were treated with @entity10 given s.c. at a dose of 3 x 10(6) units/m2 per day for 5 days per week plus weekly @entity7 given at 25 mg/m2 as an i.v. bolus. The protocol called for 4 consecutive weeks of treatment followed by 1 week off treatment. In all, 15 @entity0 had been previously treated; 6 @entity0 had failed hormonal therapy ( @entity5 ), 5 @entity0 had failed prior @entity11 treatment, and 4 @entity0 had received chemoembolization of the @entity1 and had subsequently progressed. A total of 30 @entity0 were evaluable for response. In all, 1 @entity0 (3%) achieved a partial response for 8+ months and 11 @entity0 (35%) achieved stabilization of disease. Six @entity0 had a fall in alphafetoprotein (AFP) values of > 50% during therapy. The median survival for all @entity0 was 9.5 months (range, 3-34+ months). The main side effects were hematological @entity2 and @entity4 , both of which were considered tolerable. As an indicator of the immunostimulatory effects of interferon, an elevation in serum markers of inflammation [ @entity8 ( @entity8 ), @entity9 ] was found in 15%-20% of @entity0 . All @entity0 had measurable Mx protein production during therapy, but these effects were not correlated to the clinical response. The clinical response rate achieved in this trial indicates that the combination of interferon and @entity7 , at least when used on the schedule reported herein, is not superior to treatment with either agent alone for @entity0 with advanced @entity3 . However, single @entity0 achieved a prolonged progression-free interval (8-10+ months) on this therapy, and it may therefore be an option for @entity0 who have failed prior hormonal or single-agent @entity11 therapy.\n\nTitle: No synergistic activity of @entity7 and XXXX in the treatment of @entity3 .\n\nQuestion: What is the most suitable entity to fill the 'xxxx' in the title?\nA. @entity0\nB. @entity6\nC. @entity2\nD. @entity5\nE. @entity12\nF. @entity11\nG. @entity1\nH. @entity7\nI. @entity9\nJ. @entity10\nK. @entity3\nL. @entity4\nM. @entity8",
	"answer": """Let's break down the passage and figure out the best entity for the title.

### Key Points from the Passage:
- The passage discusses the treatment of patients with advanced @entity3 using a combination of treatments, including @entity11 and @entity7.
- @entity0 refers to the patients being treated.
- The study mentions the use of alpha-interferon (likely referring to @entity11) and its combination with @entity7.
- It discusses the response rate, the treatments, and side effects, but the main point is that the combination treatment was not superior to using the agents individually for patients with advanced @entity3.

### Title:
The title suggests that there is no synergistic activity between @entity7 and another entity in treating @entity3.

### Finding the Best Fit:
The title needs to indicate what @entity7 was combined with, as the text implies it was combined with @entity11 (alpha-interferon) and discusses its performance in treating @entity3.

- @entity7 is likely the entity involved in the combination with @entity11.
- The focus of the passage is that the combination treatment did not show superior results for patients with advanced @entity3.

Thus, the entity that should be in the title to complete it meaningfully is **@entity11**, as it was the agent combined with @entity7 for treating @entity3.

### Final Answer:
The most suitable entity to fill the 'xxxx' in the title is **F. @entity11**."""
 
 }]


ddxplus_prompt = [{
    "problem": "Imagine you are a doctor, next I will give you a dialogue and please make a diagnosis decision based on it. You only need to choose one answer among the following lists: ['spontaneous pneumothorax', 'cluster headache', 'boerhaave', 'spontaneous rib fracture' 'gerd', 'hiv (initial infection)', 'anemia', 'viral pharyngitis', 'inguinal, hernia', 'myasthenia gravis', 'whooping cough', 'anaphylaxis', 'epiglottitis', 'guillain-barr\u00e9 syndrome', 'acute laryngitis', 'croup', 'psvt', 'atrial fibrillation', 'bronchiectasis', 'allergic sinusitis', 'chagas', 'scombroid food poisoning', 'myocarditis', 'larygospasm', 'acute dystonic reactions', 'localized edema', 'sle', 'tuberculosis', 'unstable angina', 'stable angina', 'ebola', 'acute otitis media', 'panic attack', 'bronchospasm / acute asthma exacerbation', 'bronchitis', 'acute copd exacerbation / infection', 'pulmonary embolism', 'urti', 'influenza', 'pneumonia', 'acute rhinosinusitis', 'chronic rhinosinusitis', 'bronchiolitis', 'pulmonary neoplasm', 'possible nstemi / stemi', 'sarcoidosis', 'pancreatic neoplasm', 'acute pulmonary edema', 'pericarditis', 'cannot decide']. The answer should be a single word. \nAge: 18 \n\nSex: M \n\nInitial evidence: Do you have a fever (either felt or measured with a thermometer)? Yes \n\nEvidence: Do you live with 4 or more people? Yes;Have you had significantly increased sweating? Yes;Do you have pain somewhere, related to your reason for consulting? Yes;Characterize your pain: Sensitive;Characterize your pain: Heavy;Do you feel pain somewhere? Forehead;Do you feel pain somewhere? Cheek(r);Do you feel pain somewhere? Temple(l);How intense is the pain? 4;Does the pain radiate to another location? Nowhere;How precisely is the pain located? 3;How fast did the pain appear? 3;Do you have a cough that produces colored or more abundant sputum than usual? Yes;Do you smoke cigarettes? Yes;Do you have a fever (either felt or measured with a thermometer)? Yes;Do you have a sore throat? Yes;Do you have a cough? Yes;Have you traveled out of the country in the last 4 weeks? N;Are you exposed to secondhand cigarette smoke on a daily basis? Yes. ",
	"answer": """Let's analyze the symptoms provided:

1. **Fever**: Yes
2. **Living with 4 or more people**: Yes (This could suggest a higher exposure to contagious diseases)
3. **Increased sweating**: Yes (Could indicate an infection or systemic issue)
4. **Pain**: Yes, with the description of the pain being **sensitive** and **heavy**. The pain is localized to the **forehead**, **cheek (right side)**, and **temple (left side)**. The intensity of the pain is rated as **4** on a scale of 1 to 10, and it doesn’t radiate to other areas.
5. **Cough**: Yes, with **colored or more abundant sputum**. This could suggest a respiratory issue, such as a bacterial infection.
6. **Sore throat**: Yes (Could be related to a viral or bacterial infection)
7. **Smoking history**: Yes (This could increase susceptibility to respiratory infections and other pulmonary conditions)
8. **Exposure to secondhand smoke**: Yes (This also increases the risk of respiratory problems)

Based on these symptoms, the possible diagnoses involve a respiratory issue with fever, pain, cough, and increased sputum production. The most likely diagnosis here could be **bronchitis**, which often presents with these symptoms, including a cough with colored sputum, fever, and pain in the forehead, cheek, and temple due to sinus congestion.

The answer is: **bronchitis**."""
}]


drugdose_prompt = [{
    "problem": "In this task, you will be given sentences and a drug name in which your task is to indicate the doses of intake of that particular drug mentioned in the sentence. A dose refers to a specified amount of medication taken at one time. You just need to extract the texts that indicate the does of the drug.\nSentence: She continued to receive regular insulin 4 times per day over the following 3 years with only occasional hives.\nDrug:  insulin",
	"answer": """Let's break down the problem step by step:

1. **Identify the drug in the sentence**:  
   The drug mentioned in the sentence is **insulin**.

2. **Identify dose-related information in the sentence**:  
   The sentence says:  
   "She continued to receive regular insulin 4 times per day over the following 3 years with only occasional hives."  
   The key information related to the dosage of insulin is **"4 times per day"**.

3. **Extract the dose information**:  
   The extracted dose-related information is **"4 times per day"**.

4. **Final answer**:  
   The dose of insulin in this sentence is **"4 times per day"**. The answer is 4 times per day"""
}]

mimic4ed_prompt = [
    {
        "problem": "age: 78\ngender: F\nn_ed_30d: 0\nn_ed_90d: 0\nn_ed_365d: 0\nn_hosp_30d: 0\nn_hosp_90d: 0\nn_hosp_365d: 0\nn_icu_30d: 0\nn_icu_90d: 0\nn_icu_365d: 0\ntriage_temperature: 97.7\ntriage_heartrate: 133.0\ntriage_resprate: 16.0\ntriage_o2sat: 94.0\ntriage_sbp: 116.0\ntriage_dbp: 77.0\ntriage_pain: 0.0\ntriage_acuity: 2.0\nchiefcom_chest_pain: False\nchiefcom_abdominal_pain: False\nchiefcom_headache: False\nchiefcom_shortness_of_breath: False\nchiefcom_back_pain: False\nchiefcom_cough: False\nchiefcom_nausea_vomiting: False\nchiefcom_fever_chills: False\nchiefcom_syncope: False\nchiefcom_dizziness: False\ncci_MI: 0\ncci_CHF: 0\ncci_PVD: 0\ncci_Stroke: 0\ncci_Dementia: 0\ncci_Pulmonary: 0\ncci_Rheumatic: 0\ncci_PUD: 0\ncci_Liver1: 0\ncci_DM1: 0\ncci_DM2: 0\ncci_Paralysis: 0\ncci_Renal: 0\ncci_Cancer1: 0\ncci_Liver2: 0\ncci_Cancer2: 0\ncci_HIV: 0\neci_Arrhythmia: 0\neci_Valvular: 0\neci_PHTN: 0\neci_HTN1: 0\neci_HTN2: 0\neci_NeuroOther: 0\neci_Hypothyroid: 0\neci_Lymphoma: 0\neci_Coagulopathy: 0\neci_Obesity: 0\neci_WeightLoss: 0\neci_FluidsLytes: 0\neci_BloodLoss: 0\neci_Anemia: 0\neci_Alcohol: 0\neci_Drugs: 0\neci_Psychoses: 0\neci_Depression: 0\nGiven the above EHR data of a patient, your goal is to predict whether the patient should be classified into critical triage queue. You should choose from the following options:\nA. Yes\nB. No",
        "answer": """To determine whether the patient should be classified into the critical triage queue, we need to evaluate the data for any signs of critical illness or risk factors that would necessitate urgent care.

### Step-by-step Analysis:

#### 1. **Demographics**
- **Age: 78**, **Gender: Female**  
  → Advanced age increases vulnerability to illness and may warrant closer monitoring.

#### 2. **Utilization History**
- All utilization metrics (**n_ed_30d, n_hosp_90d, n_icu_365d**, etc.) are **0**  
  → No recent ED visits, hospitalizations, or ICU stays. This suggests no known recent acute events.

#### 3. **Triage Vital Signs**
- **Temperature: 97.7°F** → Normal
- **Heart Rate: 133 bpm** → **Tachycardia** (high)
- **Respiratory Rate: 16** → Normal
- **Oxygen Saturation: 94%** → Borderline normal (may be low for some baseline patients, especially with comorbidities)
- **Blood Pressure: 116/77 mmHg** → Slightly low systolic; hypotension cannot be ruled out depending on baseline
- **Pain Score: 0** → No reported pain
- **Triage Acuity: 2.0** → High priority (typically reserved for patients needing immediate attention)

#### 4. **Chief Complaints**
All chief complaints are **False**, including chest pain, shortness of breath, fever, dizziness, etc.  
→ The patient does not endorse common symptoms of acute illness or distress.

#### 5. **Comorbidity Indices**
All **Charlson Comorbidity Index (CCI)** and **Elixhauser Comorbidity Index (ECI)** variables are **0**, indicating:
- No significant past medical history of chronic diseases (e.g., heart disease, cancer, diabetes, renal failure).
- This is a patient without known chronic illnesses.

### Interpretation:

Despite the lack of known chronic illness and no explicit chief complaint, several findings raise concern:
- **Tachycardia (HR = 133)** could indicate underlying infection, dehydration, cardiac pathology, or other stressors.
- **Triage acuity of 2** indicates that the triage nurse perceived this as a high-priority case requiring rapid evaluation.
- **Borderline oxygen saturation (94%)** in an elderly patient might suggest early respiratory compromise.
- **Slightly low blood pressure (116)** combined with tachycardia could indicate early shock or volume depletion.

These signs—especially **tachycardia and elevated triage acuity**—suggest the presence of a condition that may not yet have clear symptoms but has objective signs of instability.

---

### Final Decision:

Given the **objective signs of instability (tachycardia, borderline oxygenation, slightly low BP)** and **triage acuity level 2**, this patient should be considered for the **critical triage queue** for prompt evaluation and monitoring.

The answer is A. yes.
"""
    }
]

prompt_mapping = {
    'MedQA_cot': medqa_prompt,
    'pubmedqa_c_cot': pubmedqa_prompt,
    "MedMCQA_cot": medmcqa_prompt,
    "biomrc": biomrc_prompt,
    "biomrc_500": biomrc_prompt,
    "pubhealth": pubhealth_prompt,
    "medsins_task16": healthfact_prompt,
    "medsins_task130": ddxplus_prompt,
    "medsins_task29": drugdose_prompt,
    "MedMCQA_cot_500": medmcqa_prompt,
    "medsins_task130_500": ddxplus_prompt,
    "mimic4ed": mimic4ed_prompt
}