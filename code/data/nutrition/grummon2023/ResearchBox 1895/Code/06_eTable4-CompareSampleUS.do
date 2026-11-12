**********Compare Sample to US Population**********

*First, prepare US data so that demographic categories match ours 

*Bring in PUMS 2021 ACS 1-year data
/*Note: PUMS data were downloaded from the PUMS websites via FTP
*URL: https://www2.census.gov/programs-surveys/acs/data/pums/2021/1-Year/
*Then select csv_hus.zip
*These data were not uploaded to ResearchBox because they are already publicly available and because the file size is very large (230MBs)
*/

**Declare the PUMS data as "svy"
svyset [pw=pwgtp]
 
*Create appropriate age categories
rename agep age 
gen age_cat = .
	replace age_cat = 1 if inrange(age, 18,29)
	replace age_cat = 2 if inrange(age, 30, 44)
	replace age_cat = 3 if inrange(age, 45, 59)
	replace age_cat = 4 if age>=60 & age!=. 
	
	label define age_cat 1 "18 to 29 years" 2 "30 to 44 years" 3 "45 to 59 years" 4 "60 years and older"
	
	label values age_cat age_cat 
 
*Generate a sex variable to match our "female" variable
	gen female = (sex == 2)
		
*Generate a Hispanic variable to match ours
	gen hispanic = 0 if hisp==1
		replace hispanic = 1 if inrange(hisp, 2,24)
 
*Race - use categories as they are used in Table 1
	gen race_cat = . 
	replace race_cat = 4 if rac1p == 1 //White alone
	replace race_cat = 3 if rac1p == 2 //Black alone
	replace race_cat = 1 if rac1p == 3 | rac1p == 4 | rac1p == 5 // AmIndAlaskNat
	replace race_cat = 2 if rac1p == 6 | rac1p == 7 //Asian or Native Haw or Other Pacicific Islander
	replace race_cat = 5 if rac1p==8 | rac1p==9 //some other race or multiracial 
	
label define race_cat 4 "White" 3 "Black" 1 "Amer Ind Alas Native" 2 "Asian or Pacific Islander" 5 "Other or multi-racial"

label values race_cat race_cat

*Education categories - 4 levels
gen educ_cat = . 
	replace educ_cat = 1 if inrange(schl, 1,17) // HS or less
	replace educ_cat = 2 if inrange(schl, 18, 19) // some college
	replace educ_cat = 3 if inrange(schl, 20, 21) //associates or 4yr degree
	replace educ_cat = 4 if inrange(schl, 22, 24) //graduate degree
		
	label define educ_cat 1 "HS or less" 2 "Some college" 3 "2 or 4-yr degree" 4 "Grad degree"
	label values educ_cat educ_cat 
	
*Create a subpop variable to separate adults from children; we will report characteristics in the adult population only
gen subpop=.
	replace subpop=1 if age>=18 & age!=.
	replace subpop=0 if age<18 

*Calculate % in each age category and store in matrix
svy, subpop(if subpop==1): prop age_cat
	mat rtable=r(table)
	forval x=1/4 {
		mat usage_cat`x'=rtable[1,`x']
	}
*Calculate % in each gender category and store in matrix
svy, subpop(if subpop==1): prop female
	mat rtable=r(table)
	forval x=1/2 {
		mat usfemale`x'=rtable[1,`x']
	}

*Calculate % in each Hispanic category and store in matrix
svy, subpop(if subpop==1): prop hispanic
	mat rtable=r(table)
	forval x=1/2 {
		mat ushispanic`x'=rtable[1,`x']
	}

*Calculate % in each race category and store in matrix
svy, subpop(if subpop==1): prop race_cat
	mat rtable=r(table)
	forval x=1/5 {
		mat usrace_cat`x'=rtable[1,`x']
	}

*Calculate % in each education category and store in matrix
svy, subpop(if subpop==1): prop educ_cat
	mat rtable=r(table)
	forval x=1/4 {
		mat useduc_cat`x'=rtable[1,`x']
	}
	
*Store results for %s that were derived from sources other than PUMS
	
*Reports for perceived dietary quality are from doi: 10.1093/cdn/nzac067.072
//This study used NHANES 2015-2018 data
mat usdietqual1=.06
mat usdietqual2=.24
mat usdietqual3=.41
mat usdietqual4=.22
mat usdietqual5=.08

//check sums to ~100: mat dietcheck=usdietqual1[1,1]+usdietqual2[1,1]+usdietqual3[1,1]+usdietqual4[1,1]+usdietqual5[1,1]
	
**Reports for income categories are from the CPS 2021: https://www.census.gov/library/publications/2022/demo/p60-276.html
*see table on pg 16
mat usincome_cat1=0.093+0.081	
mat usincome_cat2=0.078+.109
mat usincome_cat3=.162
mat usincome_cat4=.119+.159+.083+.116
//check sums to ~100: mat incomecheck=usincome_cat1[1,1]+usincome_cat2[1,1]+usincome_cat3[1,1]+usincome_cat4[1,1]


****************************************	
*Now re-calculate characteristics for study sample and store results so we can easily output them here
****************************************

*Bring in the cleanded data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear

prop age_cat
	mat rtable=r(table)
	forval x=1/4 {
		mat sampage_cat`x'=rtable[1,`x']
	}

prop gender_cat
	mat rtable=r(table)
	forval x=1/3 {
		mat sampgender_cat`x'=rtable[1,`x']
	}

rename latino hispanic 
prop hispanic
	mat rtable=r(table)
	forval x=1/2 {
		mat samphispanic`x'=rtable[1,`x']
	}

prop race_cat
	mat rtable=r(table)
	forval x=1/5 {
		mat samprace_cat`x'=rtable[1,`x']
	}

prop educ_cat
	mat rtable=r(table)
	forval x=1/4 {
		mat sampeduc_cat`x'=rtable[1,`x']
	}
	
prop dietqual 
	mat rtable=r(table)
	forval x=1/5 {
		mat sampdietqual`x'=rtable[1,`x']
	}
	
prop income_cat
	mat rtable=r(table)
	forval x=1/4 {
		mat sampincome_cat`x'=rtable[1,`x']
	}	
	
****************************************	
*Output results to Excel
****************************************

capture rm "$Results/VLO_USvsSample.xlsx"
putexcel set "$Results/VLO_USvsSample.xlsx", replace

****
*Add column headers
****

putexcel A2=("Characteristic")
putexcel B1=("Sample")
putexcel B2=("%")
putexcel C1=("US Population")
putexcel C2=("%")

****
*Add US results
****

**Age
putexcel A3=("Age")
putexcel A4=("18-29")
putexcel A5=("30-44")
putexcel A6=("45-59")
putexcel A7=("60+")

local row=4
forval x=1/4 {
	putexcel B`row'=matrix(sampage_cat`x'), nformat(0%)
	putexcel C`row'= matrix(usage_cat`x'), nformat(0%)
	local ++row
}

**Gender
putexcel A8=("Gender")
putexcel A9=("Female")
putexcel A10=("Male")
putexcel A11=("Non-binary / another")

	*Sample
	local row=9
	foreach x in 1 2 3  { 
		putexcel B`row'= matrix(sampgender_cat`x'), nformat(0%)
		local ++row
	}

	*US
	local row=9
	foreach x in 2 1  { //opposite order so female is first, male second
		putexcel C`row'= matrix(usfemale`x'), nformat(0%)
		local ++row
	}
	putexcel C`row'=("--") //NB in NA for US sample b/c of how the ACS question is asked

**Latino(a)
putexcel A12=("Latino(a)")
putexcel B12=matrix(samphispanic2),nformat(0%) //2 is hispanic %
putexcel C12=matrix(ushispanic2), nformat(0%) //2 is hispanic %

**Race cat
putexcel A13=("Race")
putexcel A14=("American Indian")
putexcel A15=("Asian/Pac Islander")
putexcel A16=("Black or Af Am")
putexcel A17=("white")
putexcel A18=("Other or multi")

local row=14
forval x=1/5 {
	putexcel B`row'=matrix(samprace_cat`x'), nformat(0%)
	putexcel c`row'=matrix(usrace_cat`x'), nformat(0%)
	local ++row
}

**Education
putexcel A19=("Education")
putexcel A20=("HS or less")
putexcel A21=("Some college")
putexcel A22=("College grad or associates")
putexcel A23=("Graduate degree")

local row=20
forval x=1/4 {
	putexcel B`row'=matrix(sampeduc_cat`x'), nformat(0%)
	putexcel C`row'=matrix(useduc_cat`x'), nformat(0%)
	local ++row
}

**Diet quality 
putexcel A24=("Self-rated dietary quality")
putexcel A25=("Poor")
putexcel A26=("Fair")
putexcel A27=("Good")
putexcel A28=("Very good")
putexcel A29=("Excellent")

local row=25 
forval x=1/5 {
	putexcel B`row'=matrix(sampdietqual`x'), nformat(0%)
	putexcel C`row'=matrix(usdietqual`x'), nformat(0%)
	local ++row
}

**Household income
putexcel A30=("Household income")
putexcel A31=("0-24k")
putexcel A32=("25-49k")
putexcel A33=("50-74k")
putexcel A34=("75k+")

local row=31
forval x=1/4 {
	putexcel B`row'=matrix(sampincome_cat`x'), nformat(0%)
	putexcel C`row'=matrix(usincome_cat`x'), nformat(0%)
	local ++row
}

