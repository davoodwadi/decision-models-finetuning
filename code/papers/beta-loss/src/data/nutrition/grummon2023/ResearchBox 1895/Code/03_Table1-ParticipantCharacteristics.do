*********Create Table 1 - Participant Characteristics*********

/*
This do file calculates descriptive sample characteristics and outputs results to Excel. These results are presented in Table 1. 
*/

*Bring in the cleanded data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear


***
*Set the Excel file
***

cd "$Results"
capture rm "VLO_SampleChar.xlsx" //remove any old version of the file
putexcel set "VLO_SampleChar.xlsx", replace
		
***
*Add column headers
***

putexcel A2=("Characteristic")

putexcel B1=("Control (Calorie labels)")
putexcel D1=("Green labels")
putexcel F1=("Traffic light labels")
putexcel H1=("Physical activity equivalent labels")
putexcel J1=("High in warning labels")

putexcel B2=("N")
putexcel C2=("Percent")

putexcel D2=("N")
putexcel E2=("Percent")

putexcel F2=("N")
putexcel G2=("Percent")

putexcel H2=("N")
putexcel I2=("Percent")

putexcel J2=("N")
putexcel K2=("Percent")
	
***
*Add N's and %s for each of the 5 labeling conditions using a loop over foplcond
***

*First, set the starting columns for the Ns and %s
local x = 2 //col1 - start in B 
local y = 3 //col2 - start in C
		
*Loop over the 5 conditions (can change order if desired)
foreach condition in 1 2 3 4 5 { 

*Age
	putexcel A3=("Age")
	
	local row=4
	local col1: word `x' of `c(alpha)' 
	local col2: word `y' of `c(alpha)' 
	
	foreach var in age_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}
	 
	}

*Gender
 	putexcel A`row'=("Gender identity")
	foreach var in gender_cat {
			tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}
	}
	

*Latino or Hispanic 
	putexcel A`row'=("Hispanic ethnicity")
	foreach dichotvar in latino {
		count if `dichotvar' == 1 & foplcond==`condition' 
			matrix `dichotvar'n=r(N)
			putexcel `col1'`row'=matrix(`dichotvar'n)
			
		prop `dichotvar' if foplcond==`condition'
		putexcel `col2'`row'=matrix(100*(`dichotvar'n/e(N))), nformat("(##)")
		local ++row	
	}

*Race
	putexcel A`row' = ("Race")
	
	foreach var in race_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}
	}

*Education
	putexcel A`row' = ("Education")
	
	foreach var in educ_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}
	}
	
*NFP Use
	putexcel A`row' = ("Use of NF label")
	
	foreach var in NFPuse {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}

*Self-rated diet quality
	putexcel A`row' = ("Self-rated diet quality")
	
	foreach var in dietqual {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}

*Household size
	putexcel A`row' = ("Household size")
	
	foreach var in hhsize_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}

*Number of children
	putexcel A`row' = ("Number of children")
	
	foreach var in numchildren_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}

*Annual household income - 4 category variable
	putexcel A`row' = ("Household income, annual")
	
	foreach var in income_cat {
		tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}
	
*Income <=150% FPL
	putexcel A`row'=("Income <=150% Federal Poverty Level")
	foreach dichotvar in lowincome {
		count if `dichotvar' == 1 & foplcond==`condition' 
			matrix `dichotvar'n=r(N)
			putexcel `col1'`row'=matrix(`dichotvar'n)
			
		prop `dichotvar' if foplcond==`condition'
		putexcel `col2'`row'=matrix(100*(`dichotvar'n/e(N))), nformat("(##)")
		local ++row	
	}
	
*BMI category
	putexcel A`row'=("Body Mass Index category")
	foreach var in bmi_cat1 { 
	tabulate `var' if foplcond==`condition', matcell(`var'freq) matrow(`var'names)
			local ++row
			putexcel A`row'= matrix(`var'names) 
			putexcel `col1'`row'=matrix(`var'freq) 
			putexcel `col2'`row'=matrix(100*(`var'freq/r(N))), nformat("(##)") 
		
			local rows = rowsof(`var'names)
			forval i=1/`rows'   {
				local val = `var'names[`i',1]
				local val_lab : label (`var') `val'
				putexcel A`row'=("`val_lab'")
				local ++row
				di "row is `row'"
			}	 
	}
	
local x=`x'+2 // increment column up 2 (N and %)
local y=`y'+2 // increment column up 2 (N and %)
 
}


*Estiamte amount of missing data across demo's reported in Table 1
global table1vars age_cat gender_cat latino race_cat educ_cat NFPuse dietqual hhsize_cat numchildren_cat income_cat lowincome 


foreach variable in $table1vars {
	mdesc `variable'
	
}


