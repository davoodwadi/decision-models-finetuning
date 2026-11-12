/*Data Prep*/

*Bring in the survey data
import delimited "$Data/VLO-Survey-2022-3+FINAL_December+5,+2022_09.36-NoIDs.csv", case(preserve) clear 

*Drop ineligible participants per pre-registration
drop if svy_status!="complete" //did not complete survey
drop if Q_TotalD<$toofast //completed too quickly
drop if confirmkeepdata==0 //asked for data to be removed in debrief
drop if consent==0 //did not consent
drop if age<18 //under age 18, so not eligible

************************
*Initial Clean up      *
************************

*Housekeeping - decode missings, destring as needed
mvdecode *, mv(-99)
destring foplcond, replace
destring exptorder, replace

*Create a more intuitive version of the FOPL condition variable that uses abbreviated condition names
gen FOPLname = ""
replace FOPLname="Cal" if foplcond==1
replace FOPLname="G" if foplcond==2
replace FOPLname="TLL" if foplcond==3
replace FOPLname="PA" if foplcond==4
replace FOPLname="W" if foplcond==5

*Label the numeric FOPL condition variable
label define foplcond 1 "Calorie" 2 "Green only" 3 "TLL" 4 "PA Equiv" 5 "High In Warning" 
label values foplcond foplcond

*Create a PID variable
gen pid=_n

************************************************
*Merge variables across FOPL condtions         *
************************************************
	
**Combine the five FOPL condition product selection variables into a merged variable for each task (beverages and snacks)
foreach category in bev snack {
	gen `category'_choice_merged=.
	
	foreach cond in Cal G TLL PA W {
		replace `category'_choice_merged = `category'_choice_`cond' if FOPLname=="`cond'"
	}
	*Drop un-merged variables - no longer needed
	drop `category'_choice_Cal `category'_choice_G `category'_choice_TLL `category'_choice_PA `category'_choice_W
}

*Check that there is not missing data on choice variable (should be minimal or 0 missingness given we prompted responses)
assert bev_choice_merged!=0
assert snack_choice_merged!=0

*Create a variable equal to the item number selected in each of the two choice tasks. We will use this to merge selection data to product characteristic data. 
gen itemnumber_bev = bev_choice_merged
gen itemnumber_snack = snack_choice_merged

************************************************************
*Merge selection data with product characteristics         *
************************************************************

*Merge product choices from the survey data with product characteristics using itemnumber as the linking variable
*First, merge with beverage characteristics database
merge m:1 itemnumber_bev using "$Data/BeverageCharacteristics.dta"
	rename _merge bevmerge
	*Drop products never selected (_merge==2 - i.e., in using (ProductChar) data only)
	drop if bevmerge==2
	
	*Check that there are no products in master data only (this would indicate selection of a product not in the ProductChar database)
	assert bevmerge!=1 if itemnumber_bev!=. 

*Next, merge with snack characteristics database
merge m:1 itemnumber_snack using "$Data/IntermediateData/SnackCharacteristics.dta"
	rename _merge snackmerge
	*As above, drop products never selected (_merge==2 - using data only)
	drop if snackmerge==2
	
	*Check that there are no products in master data only
	assert snackmerge!=1 if itemnumber_snack!=.
	
*****************************************************
*Create additional selection variables for analysis *
*****************************************************

*Create a numeric version of the "RYG" variable, which is the variable for selection of red (unhealthy), yellow (moderately healthy) or green (healthy) item
foreach catg in bev snack {
	capture rename item_col_str_jo_`catg' RYG_`catg'
	gen RYG_`catg'_num = .
	replace RYG_`catg'_num = 1 if RYG_`catg'=="G"
	replace RYG_`catg'_num = 2 if RYG_`catg'=="Y"
	replace RYG_`catg'_num = 3 if RYG_`catg'=="R"
	
}

label define RYG_num 1 "Green" 2 "Yellow" 3 "Red"
label values RYG_snack_num RYG_bev_num RYG_num

*rename the variable item_col_str_jo_`catg' as RYG_`catg' (more intuitive to me); loop this over beverages and snacks
foreach catg in bev snack {		
	gen green`catg' = .
	replace green`catg' = 1 if RYG_`catg'=="G"
	replace green`catg' = 0 if RYG_`catg'=="Y" | RYG_`catg'=="R"
	
	gen yellow`catg'=.
		replace yellow`catg' = 1 if RYG_`catg'=="Y"
		replace yellow`catg' = 0 if RYG_`catg'=="G" | RYG_`catg'=="R"
		
	gen red`catg'=.
		replace red`catg' = 1 if RYG_`catg'=="R"
		replace red`catg' = 0 if RYG_`catg'=="G" | RYG_`catg'=="Y"
}

	
*******************************************************
*Finish merge variables across FOPL condtions         *
*******************************************************

*Fix a few variable names that used different capitlization/case for FOPL conditions in the Qualtrics survey (e.g., Cal instead of CAL)
rename stigma_personal_CAL stigma_personal_Cal
rename stigma_obesity_CAL stigma_obesity_Cal 
rename stigma_blame_CAL stigma_blame_Cal
rename disgust_CAL disgust_Cal

*Fix coding of response options for snack_healthyY_W, which was miscoded in response option coding on the Qualtrics survey. Need to replace 10-->5. (See codebook for confirmation this is the correct coding. Not that numbers were not shown to participants, so it is appropriate to recode 10->5 here). 
replace snack_healthyY_W = 5 if snack_healthyY_W==10

*For remaining variables, create merged variable that combines responses across the 5 FOPL conditions
foreach v in bev_healthyR bev_healthyY bev_healthyG snack_healthyR snack_healthyY snack_healthyG noticed_cal noticed_trial infl_bev help_bev infl_snack help_snack attn elab pme_disc pme_enc affect_worried affect_scared affect_guilty affect_ashamed affect_sad learn trust control react_manip react_annoy stigma_blame stigma_personal stigma_obesity disgust   { 
	gen `v'_merged = . 
	foreach cond in Cal G TLL PA W {
		replace `v'_merged = `v'_`cond' if FOPLname=="`cond'"
	}
	*Drop un-merged variables - no longer needed
	drop `v'_Cal `v'_G `v'_TLL `v'_PA `v'_W
}


************************************************
*Prep and Label Beverage/Snack Perceptions     *
************************************************

*Label response optoins for healthfulness perceptions variables 
label define healthy 1 "Very unhealthy" 2 "Somewhat unhealthy" 3 "Neither unhealthy nor healthy" 4 "Somewhat healthy" 5 "Very healthy"
label values bev_healthyR_merged bev_healthyY_merged bev_healthyG_merged snack_healthyR_merged snack_healthyY_merged snack_healthyG_merged healthy


************************************************
*Prep and Label the Label Reaction Variables   *
************************************************

*Label response options for noticing variables
label define noticed 1 "Yes" 0 "No" 2 "Not sure"
label values noticed_cal_merged noticed_trial_merged noticed 

*Create a binary variable for 1=yes noticed (and 0=no or not sure)
foreach label in cal trial { 
	gen yesnoticed_`label'_merged = .
	replace yesnoticed_`label'_merged = 1 if noticed_`label'_merged==1
	replace yesnoticed_`label'_merged = 0 if noticed_`label'_merged==0 | noticed_`label'_merged==2
}

*Label the influenced variables
label define influenced 1 "Yes" 0 "No" 2 "Did not see label"
label values infl_bev infl_snack influenced

*Create a binary variable for label influenced choice where 1=yes noticed and 0=no or not sure
foreach product in bev snack {
	gen yesinfl_`product' = 1 if infl_`product'==1
	replace yesinfl_`product'=0 if infl_`product'==0 | infl_`product'==2
}

*Define label for "Not at all" to "Quite a bit" (response options that are used often)
label define howmuch5ptlikert 1 "Not at all" 2 "Very little" 3 "Somewhat" 4 "Quite a bit" 5 "A great deal"

*Label variables with "Not at all" to "Quite a bit" response options
label values help_bev_merged help_snack_merged attn_merged elab_merged pme_disc_merged pme_enc_merged affect_worried_merged affect_scared_merged affect_guilty_merged affect_ashamed_merged affect_sad_merged learn_merged react_manip_merged react_annoy_merged  stigma_personal_merged stigma_obesity_merged stigma_blame_merged disgust_merged howmuch5ptlikert

*Define response option labels for trust in label variable and apply to responses
label define trust 1 "Completely distrust" 2 "Somewhat distrust" 3 "Neither trust/distrust" 4 "Somewhat trust" 5 "Completely trust"
label values trust trust 

*Label values for perceived control
label define perceivedcontrol 1 "Decreased control" 2 "Neither decrease/increase" 3 "Increased control"
label values control_merged perceivedcontrol 

*Create variable for increased perceived control (vs. same or decreased)
	gen increase_control = .
		replace increase_control = 1 if control_merged==3 //increased
		replace increase_control = 0 if inrange(control_merged,1,2) //decreased or neither
		
***
*Create scales for negative affect/emotions, reactance, perceived obesity stigma 
***

*Negative affect / emotions
	egen affect_avg = rowmean(affect_worried_merged affect_scared_merged affect_guilty_merged affect_ashamed_merged affect_sad_merged)
	
*Reactance
	egen react_avg = rowmean(react_manip_merged react_annoy_merged)
	
*Perceived personal stigma
	egen obesitystigma_avg = rowmean (stigma_blame_merged stigma_obesity_merged)

***
*Standardize continuous label reaction DVs for ease of comparison across outcomes
***

foreach variable in $continvars {
	egen z`variable' = std(`variable')
}


*******************************************
*Prep and Label Demographic Variables     *
*******************************************

*Create age categories 
gen age_cat = .
	replace age_cat = 1 if inrange(age, 18,29)
	replace age_cat = 2 if inrange(age, 30, 44)
	replace age_cat = 3 if inrange(age, 45, 59)
	replace age_cat = 4 if age>=60 & age!=.
	
	label define age_cat 1 "18 to 29 years" 2 "30 to 44 years" 3 "45 to 59 years" 4 "60 years and older"
	label values age_cat age_cat
	
*Gender
gen gender_cat = . 
	replace gender_cat = 1 if gender==1 
	replace gender_cat = 2 if gender==2
	replace gender_cat = 3 if gender==3 | gender==4 
	
	label define gender_cat 1 "Woman" 2 "Man" 3 "Non-binary or another gender"
	label values gender_cat gender_cat 
	
*Woman/Man
	gen woman = .
		replace woman=1 if gender==1 
		replace woman=0 if gender==2
		//leave as missing for NB and another gender

*Race categories
	*Rename race variable to be lower case
	forval x=1/6 {
		rename Race_`x' race_`x'
	}
	
	gen race_cat = . 
		replace race_cat = 1 if race_1==1 & race_2==0 & race_3==0 & race_4==0 & race_5==0 & race_6==0 //AmInd only
		replace race_cat = 2 if (race_2==1 | race_4==1) & race_1==0 & race_3==0 & race_5==0 & race_6==0 //Asian only 
		replace race_cat = 3 if race_3==1 & race_1==0 & race_2==0 & race_4==0 & race_5==0 & race_6==0 //Black only
		replace race_cat = 4 if  race_5==1 & (race_1==0 & race_2==0 & race_3==0 & race_6==0) //White only
		replace race_cat = 5 if race_6 ==1 //Other 
		
		*count total races marked and replace those who marked more than 1 race as "other or mutliracial"
		egen totalraces = rowtotal(race_1 race_2 race_3 race_4 race_5 race_6)
		replace totalraces = 1 if totalraces==2 & race_4==1 & race_2==1 
		replace race_cat = 5 if totalraces>=2 & totalraces!=.
	
	label define race_cat 1 "Amer Ind Alas Native" 2 "Asian or Pacific Islander" 3 "Black" 4 "White" 5 "Other or multi-racial"
	label values race_cat race_cat
		
*Education
	gen educ_cat = .
		replace educ_cat = 1 if inrange(educ, 1, 2) //HS or less
		replace educ_cat = 2 if educ==3 //some college
		replace educ_cat = 3 if inrange(educ, 4, 5) //associates or bachelors
		replace educ_cat = 4 if educ==6 //graduate or professional degree
		
	label define educ_cat 1 "HS or less" 2 "Some college" 3 "2 or 4-yr degree" 4 "Grad degree"
	label values educ_cat educ_cat 
	
*Generate indicator for "low education" = some college or less, for pre-specified moderation analyses
	gen loweduc = .
		replace loweduc = 1 if inrange(educ_cat, 1, 2)
		replace loweduc = 0 if inrange(educ_cat, 3, 4)

*Label diet quality variable
	label define dietqual 1 "Poor" 2 "Fair" 3 "Good" 4 "Very good" 5 "Excellent"
	label values dietqual dietqual

*Create variable for whether the numeracy question was answered correctly or not
	gen numeracy_correct = .
		replace numeracy_correct = 1 if numeracy==25
		replace numeracy_correct = 0 if numeracy!=25 & numeracy!=.

*Label health literacy values - how often need help reading  medical information
	label define healthlit 1 "Never" 2 "2" 3 "Sometimes" 4 "4" 5 "Always" //points 2 and 4 are not labeled in survey
	label values healthlit healthlit

*Label "How often do you use the Nutrition Facts Panel" variable 
	label define NFPuse 1 "Never" 2 "Rarely" 3 "Sometimes" 4 "Most of the time" 5 "Always"
	label values NFPuse NFPuse
	
	
*Label vending frequency variables
	label define vendingfreq 1 "Never or <1 time/week" 2 "1-2 times/week" 3 "3-4 times/week" 4 "5+ times/week"
	label values vendingfreq_bev vendingfreq_snack vendingfreq
	
*Generate a 4-level collapsed political party variable
	gen politparty = .
		replace politparty = 1 if inrange(partyid, 1, 3) //democrat
		replace politparty = 2 if inrange(partyid, 5, 7) //republican
		replace politparty = 3 if partyid==4 // independent
		replace politparty = 4 if partyid==8 //something else
	
	label define politparty 1 "Democrat" 2 "Republican" 3 "Independent" 4 "Something else"
	label values politparty politparty

*Create a 4-level collapsed variable for household size categories 
	gen hhsize_cat = . 
		replace hhsize_cat = 1 if hhsize_num==1
		replace hhsize_cat = 2 if hhsize_num==2
		replace hhsize_cat = 3 if hhsize_num==3
		replace hhsize_cat = 4 if inrange(hhsize_num,4,20)
		
	label define hhsize_cat 1 "1" 2 "2" 3 "3" 4 "4 or more"
	label values hhsize_cat hhsize_cat 
	
*Create a 4-level collapsed variable for number of children categories
	gen numchildren_cat = . 
		replace numchildren_cat = 1 if hhsize_num==1 & children_num==. //replace num children as 0 if HH size is 1 and number of children is missing
		replace numchildren_cat = 1 if children_num==0
		replace numchildren_cat = 2 if children_num==1
		replace numchildren_cat = 3 if children_num==2
		replace numchildren_cat = 4 if inrange(children_num,3,15)
		
	label define numchildren_cat 1 "0" 2 "1" 3 "2" 4 "3 or more"
	label values numchildren_cat numchildren_cat 
	
*Create height in inches, combining from height in feet and height in inches
gen total_height_inches = .
	replace total_height_inches = (height_feet*12) + height_inch
	
*Convert height to meters for BMI calculation
	gen height_m = total_height_inches * 0.0254
	
*Create a m-squared variable for easy BMI calculation
	gen heightsq_m2 = height_m * height_m
	
*Convert weight to kg for BMI calculation
	gen weight_kg = weight_lbs * 0.453592
	
*Create a BMI variable = kg/m2
	gen bmi = (weight_kg/heightsq_m2) if weight_kg!=. & heightsq!=.
		replace bmi=. if weight_kg==. | heightsq_m2==.

*Create a variable for having BMI in 'obese' category
	gen obesebmi = .
		replace obesebmi = 1 if bmi >=30.0 & bmi !=.
		replace obesebmi=0 if bmi<30 & bmi!=.

*Create a BMI category variable; keep missing values as system missing 
	gen bmi_cat1 = .
		replace bmi_cat1 = 1  if bmi < 18.5 // underweight
	replace bmi_cat1 = 2 if bmi >=18.5 & bmi < 25 //healthy 
	replace bmi_cat1 = 3 if bmi >= 25 & bmi < 30 //overweight
	replace bmi_cat1 = 4 if bmi > 30 & bmi !=. //obese
	
	label define bmi_cat1 1 "Underweight" 2 "Healthy" 3 "Overweight" 4 "Obese" 
	label values bmi_cat1 bmi_cat1

*Label the 10 category annual household income variable
	label define income_10cat 1 "<$10,000" 2 "$10,000-14,999" 3 "$15,000-24,999" 4 "$25,000-34,999" 5 "$35,000-49,999" 6 "$50,000-74,999" 7 "$75,000-99,999" 8 "$100,000-149,999" 9 "$150,000-199,999" 10 "$200,000 or more"
	label values income_10cat income_10cat
	
*Create a collapsed version of the annual HH income variable (4 levels)
	gen income_cat = .
		replace income_cat = 1 if inrange(income_10cat, 1, 3) //LT 25k
		replace income_cat = 2 if inrange(income_10cat, 4, 5) //25-49k
		replace income_cat = 3 if income_10cat==6 //50-74k
		replace income_cat = 4 if income_10cat>=7 & income_10cat!=. //>=75k
		
	label define income_cat 1 "0-24,999" 2 "25,000 to 49,999" 3 "50,000 to 74,999" 4 "75,000+"
	label values income_cat income_cat 
	
*Create a variable for income less than or equal to 150% FPL (i.e., response to income as FPL question was "1")
	gen lowincome = .
		forval hhsize=1/20 {
			replace lowincome = 1 if fpl_`hhsize' == 1 
			replace lowincome = 0 if fpl_`hhsize' >1 & fpl_`hhsize' !=.
		}	

*Save cleaned dataset
save "$Data/VLOData_clean.dta", replace
