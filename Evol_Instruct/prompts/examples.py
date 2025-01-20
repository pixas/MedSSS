mcts_example = [
    "<problem>\nA 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin\n</problem>\n\n<steps>\nStep0: Let's break down this proble step by step.\n\nStep1: Identify the likely diagnosis (UTI vs other conditions). The patient presents with symptoms of burning upon urination, which is a common symptom of a urinary tract infection (UTI). However, other conditions such as vaginitis or interstitial cystitis could present with similar symptoms. The absence of fever and costovertebral tenderness helps rule out pyelonephritis (upper UTI), making the diagnosis of cystitis (lower UTI) more likely.\n\nStep2: Assess appropriate treatment options for UTIs in pregnancy. Determine which antibiotics are considered safe for use in pregnancy to treat UTIs. The treatment options should be effective against the most common pathogens, typically Escherichia coli (E. coli). In pregnancy, the choice of antibiotics is guided by safety for both the mother and fetus.\n\nStep 3: Consider safety for both mother and fetus when choosing an antibiotic. Nitrofurantoin is commonly used for the treatment of UTIs in pregnancy, as it is considered safe for use during the second trimester. However, nitrofurantoin should be avoided in the first trimester due to concerns about teratogenicity, and it is not recommended in the third trimester due to the potential risk of hemolytic anemia in the newborn.\n\nStep 4: Given the patient's symptoms, the diagnosis of cystitis is most likely, and nitrofurantoin can be prescribed, provided the patient is in the second trimester of pregnancy. The answer is D.\n</steps>"
#     """<problem>\nA 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin\n</problem>\n\n<steps>\nStep0: Let's break down this proble step by step.\n\nStep 1: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination, which suggests a urinary tract infection (UTI).

# Step 2: The absence of costovertebral angle tenderness makes pyelonephritis less likely, so it's probably a lower UTI, like cystitis.

# Step 3: The patient has already tried increasing her water intake and taking cranberry extract, but it's not helping, so she needs some treatment.

# Step 4: The options for antibiotic treatment are Ampicillin, Ceftriaxone, Doxycycline, and Nitrofurantoin. The goal is to find a safe and effective option for a pregnant woman with a UTI.

# Step 5: Ampicillin is considered safe in pregnancy and is often used for bacterial infections, including UTIs. Ceftriaxone is also safe but might be reserved for more severe infections or if there's penicillin allergy.

# Step 6: Doxycycline is not recommended in pregnancy due to potential harm to the fetus. Nitrofurantoin is considered relatively safe and is frequently used for UTIs in pregnancy, especially in the first and second trimesters.

# Step 7: However, nitrofurantoin may not be the best option for the patient since she's at 22 weeks gestation, and there are other considerations to take into account, such as local resistance patterns and potential side effects.

# Step 8: The American College of Obstetricians and Gynecologists (ACOG) recommends treating UTIs in pregnancy to prevent complications like pyelonephritis, and nitrofurantoin is generally safe for treating uncomplicated UTIs in pregnancy.

# Step 9: Considering the patient's symptoms, which include burning upon urination typical of cystitis, and the fact that there are no signs of pyelonephritis, treating with nitrofurantoin should be sufficient.

# Step 10: However, the patient's local resistance patterns and potential allergies to medications should also be considered.

# Step 11: Nitrofurantoin is a more targeted antibiotic with fewer systemic side effects compared to ampicillin and is less likely to cause disruption of gut flora, making it a better option for this patient.

# Step 12: Therefore, nitrofurantoin is the best choice for this patient. The answer is D.</steps>"""
]

mcts_refine_example = [
    "<problem>\nA 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin\n</problem>\n\n<steps>\nStep0: Let's break down this proble step by step.\n\nStep1: Identify the likely diagnosis (UTI vs other conditions). The patient presents with symptoms of burning upon urination, which is a common symptom of a urinary tract infection (UTI). However, other conditions such as vaginitis or interstitial cystitis could present with similar symptoms. The absence of fever and costovertebral tenderness helps rule out pyelonephritis (upper UTI), making the diagnosis of cystitis (lower UTI) more likely.\n\nStep2: A likely treatment for a urinary tract infection in pregnancy is to recommend over-the-counter cranberry extract, as it is known to help prevent and treat UTIs by acidifying the urine. Additionally, increasing fluid intake, as the patient has already been doing, should resolve the infection without the need for antibiotics.\n\n<reflect>\nStep 3: Reflect: Cranberry extract has been shown to reduce the frequency of UTIs, but it is not an effective treatment for an active UTI, especially during pregnancy. The patient's symptoms have worsened despite cranberry use, indicating the need for a more direct intervention. The proper course of action would be to prescribe a pregnancy-safe antibiotic, such as nitrofurantoin or amoxicillin, after confirming the diagnosis with a urine culture. It is important to avoid self-treatment with cranberry extract or solely increasing fluid intake when dealing with a confirmed UTI in pregnancy, as it may not sufficiently address the bacterial infection.</reflect></steps>"
]

mcts_plan_examples = [
    """Problem: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?
A. Ampicillin
B. Ceftriaxone
C. Doxycycline
D. Nitrofurantoin\nPlan: \nPlan 1:
1. Gather Patient History: Review the patient's symptom onset, duration, and any previous urinary issues.
2. Consider Risk Factors: Identify factors that might predispose her to a UTI, particularly during pregnancy.
3. Evaluate Treatment Options: Analyze the safety and efficacy of available antibiotics for treating a UTI in pregnant women.

Plan 2: 
1. List Available Antibiotics: Identify the options provided in the question.
2. Assess Each Option: Evaluate each antibiotic for its effectiveness against UTIs and safety profile in pregnancy.
3. Determine Best Practice: Reference guidelines for treating UTIs in pregnant patients to guide the treatment choice.

Plan 3:
1. Focus on Symptoms: Analyze the primary symptom of burning urination and any accompanying symptoms.
2. Consider Differential Diagnosis: Rule out other potential causes of burning urination, such as vaginal infections.
3. Select Empirical Treatment: Choose an appropriate antibiotic based on the most likely diagnosis and its safety in pregnancy."""
]


mcts_ls_examples = [
    """Problem: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin\nOutput: Alright, so I'm faced with this medical question. Let's break it down step by step. We've got a 23-year-old pregnant woman at 22 weeks gestation who's presenting with burning upon urination. She says it started just one day ago and has been getting worse despite her drinking more water and taking cranberry extract. She feels well otherwise and is being followed by a doctor for her pregnancy. Her vital signs are all within normal limits: temperature is 97.7°F, blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam doesn't show any costovertebral angle tenderness, and there's a gravid uterus.

So, the main issue here is burning upon urination, which suggests a urinary tract infection (UTI). UTIs are pretty common, especially in women, and pregnant women are at an increased risk. The absence of costovertebral angle tenderness makes pyelonephritis less likely, so probably a lower UTI, like cystitis.

She’s already tried increasing her water intake and taking cranberry extract, but it's not helping. Now, she needs some treatment.

Looking at the options:

A. Ampicillin
B. Ceftriaxone
C. Doxycycline
D. Nitrofurantoin

I need to consider which antibiotic is safe and effective for a pregnant woman with a UTI.

First, Ampicillin. It's a broad-spectrum penicillin antibiotic. It's generally considered safe in pregnancy, but is it the first choice for UTIs?

Ceftriaxone is a third-generation cephalosporin. It's also considered safe in pregnancy, but again, is it the best choice for a UTI?

Doxycycline is a tetracycline antibiotic. I recall that tetracyclines are generally avoided in pregnancy, especially during the second and third trimesters, because they can affect fetal bones and teeth.

Nitrofurantoin is commonly used for urinary tract infections. I think it's category B for pregnancy, meaning it's considered relatively safe.

But wait, I should double-check the safety of these antibiotics during pregnancy.

Ampicillin is considered safe and is often used in pregnancy for bacterial infections, including UTIs. Ceftriaxone is also safe but might be reserved for more severe infections or if there's penicillin allergy.

Doxycycline is not recommended in pregnancy due to potential harm to the fetus.

Nitrofurantoin is frequently used for UTIs in pregnancy, especially in the first and second trimesters. However, it's not typically used in the third trimester because it can cause hemolysis in newborns with glucose-6-phosphate dehydrogenase deficiency and may lead to pulmonary toxicity in the fetus if delivered near term.

Given that she's at 22 weeks gestation, nitrofurantoin could be an option, but I need to consider if there are better alternatives.

Also, I should think about the guidelines for treating UTIs in pregnancy. The American College of Obstetricians and Gynecologists (ACOG) recommends treating UTIs in pregnancy to prevent complications like pyelonephritis.

Looking up the ACOG guidelines, they suggest that nitrofurantoin is generally safe for treating uncomplicated UTIs in pregnancy, especially in the first two trimesters.

Ampicillin is also a good option, especially if there's concern about resistance to nitrofurantoin.

Ceftriaxone is more potent and is usually reserved for more severe infections or if the bacteria are resistant to other antibiotics.

Doxycycline, as mentioned, is not recommended in pregnancy.

Considering all this, nitrofurantoin seems like a reasonable choice for an uncomplicated UTI in a pregnant woman in the first or second trimester.

But I should also think about the patient's symptoms. She has burning upon urination, which is typical for cystitis. No fever, no costovertebral angle tenderness, and normal vital signs, so no signs of pyelonephritis.

Given that, treating with an antibiotic like nitrofurantoin should be sufficient.

However, I should consider the local resistance patterns. If there's high resistance to nitrofurantoin, then another antibiotic might be preferred.

Also, I need to ensure that the patient doesn't have any allergies to these medications.

Assuming no allergies and no resistance issues, nitrofurantoin is a good option.

But wait, ampicillin is also a possibility. So, why choose one over the other?

Nitrofurantoin is more targeted towards urinary tract pathogens and has fewer systemic side effects compared to ampicillin.

Plus, it's less likely to cause disruption of gut flora, which is a bonus.

Ampicillin is broad-spectrum and might cover a wider range of bacteria, but for a typical UTI caused by E. coli, nitrofurantoin is sufficient.

Also, considering the pregnancy aspect, nitrofurantoin is category B and is widely used in pregnancy for UTIs.

Therefore, nitrofurantoin seems like the best choice here.

The answer is D."""
]