******Impact of FOPLs on Product Perceptions and Label Reactions*****

/**************
This do file:
-Examines how FOPL arms impact psychological outcomes (perceptions + reactions), including estimating average differential effects (ADEs) for each intervention arm vs. control and calculating p-values for significance of these differences vs. control
-Compares the ADEs to one another, including calculating p-values for differences between ADEs
-Adjusts p-values for multiple testing using Bonferroni Holm method and the qqvalue package
-Outputs these ADEs  and p-values to Excel. These form basis for Supplemental Table 8 and for any discussion of which labeling arms differed from one another in their impacts on secondary psych outcomes
**************/

*Bring in the cleanded data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear


******************
*Continuous variables
******************

 foreach dv in $continvars { 
	reg z`dv' i.foplcond
	mat rtable = r(table)
	
	*Adjust pvalues using Bonferroni Holm and store
	test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
	mat ps=r(mtest)
	
	*Green 
	mat `dv'2b = rtable[1,2] 
	mat `dv'2lb = rtable[5,2] 
	mat `dv'2ub = rtable[6,2] 	
	mat `dv'2adjp= ps[1,4]
	
	*TLL
	mat `dv'3b = rtable[1,3] 
	mat `dv'3lb = rtable[5,3] 
	mat `dv'3ub = rtable[6,3] 	
	mat `dv'3adjp= ps[2,4]
	
	*PACE
	mat `dv'4b = rtable[1,4] 
	mat `dv'4lb = rtable[5,4] 
	mat `dv'4ub = rtable[6,4] 
	mat `dv'4adjp= ps[3,4]
	
	*Warn 
	mat `dv'5b = rtable[1,5] 
	mat `dv'5lb = rtable[5,5] 
	mat `dv'5ub = rtable[6,5] 
	mat `dv'5adjp= ps[4,4]
	
	
	*Test ADEs against one another, store p-values for later adjustment 
	reg z`dv' i.foplcond	
		*Green vs. others
		forval x=3/5 {
			lincom 2.foplcond-`x'.foplcond
			mat `dv'2vs`x'p=r(p)
		}
		
		*TLL vs others
		forval x=4/5 {
			lincom 3.foplcond-`x'.foplcond
			mat `dv'3vs`x'p=r(p)
		}
		
		*PACE vs others
		foreach x in 5 {
			lincom 4.foplcond-`x'.foplcond
			mat `dv'4vs`x'p=r(p)
		}
}

	
******************
*Binary outcomes
******************

foreach dv in $binvars {
logit `dv' i.foplcond 
	margins, dydx(foplcond) post 
	mat rtable=r(table)
	*Adjust pvalues using Bonferroni Holm and store
	test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
	mat ps=r(mtest)
	
	*Green 
	mat `dv'2b = rtable[1,2] 
	mat `dv'2lb = rtable[5,2] 
	mat `dv'2ub = rtable[6,2] 	
	mat `dv'2adjp= ps[1,4]

	
	*TLL
	mat `dv'3b = rtable[1,3] 
	mat `dv'3lb = rtable[5,3] 
	mat `dv'3ub = rtable[6,3] 	
	mat `dv'3adjp= ps[2,4]
	
	*PACE
	mat `dv'4b = rtable[1,4] 
	mat `dv'4lb = rtable[5,4] 
	mat `dv'4ub = rtable[6,4] 
	mat `dv'4adjp= ps[3,4]
	
	*Warn 
	mat `dv'5b = rtable[1,5] 
	mat `dv'5lb = rtable[5,5] 
	mat `dv'5ub = rtable[6,5] 
	mat `dv'5adjp= ps[4,4]
	
	
	*Test ADEs against one another, store p-values for later adjustment 
	logit `dv' i.foplcond
	margins, dydx(foplcond) post coefleg 
		*Green vs. others
		forval x=3/5 {
			lincom  _b[2.foplcond]-_b[`x'.foplcond]
			mat `dv'2vs`x'p=r(p)
		}
		
		*TLL vs others
		forval x=4/5 {
			lincom _b[3.foplcond]-_b[`x'.foplcond]
			mat `dv'3vs`x'p=r(p)
		}
		
		*PACE vs others
		foreach x in 5 {
			lincom _b[4.foplcond]-_b[`x'.foplcond]
			mat `dv'4vs`x'p=r(p)
		}

}


/**************************************************
 Adjust p-values to q-values using Bonferroni-Holm
***************************************************/

*First, output p-values to excel
capture rm "$Results/PerceptReactCompareFOPLs_ps.xlsx"
putexcel set "$Results/PerceptReactCompareFOPLs_ps.xlsx", replace

putexcel A1=("outcome")
putexcel B1=("comparison")
putexcel C1=("pvalue")

local row=2 
foreach outcome in $continvars $binvars {
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		putexcel A`row'=("`outcome'")
		putexcel B`row'=("`comp'")
		putexcel C`row'=matrix(`outcome'`comp'p)	
		local ++row
	}
}

preserve
*Now, import the excel file created w/ the pvalues and use these data to calculate q-values using Hochberg
import excel "$Results/PerceptReactCompareFOPLs_ps.xlsx", sheet("Sheet1") firstrow clear

*Calc qvalues, adjusting using Bonferroni Holm method 
bysort outcome: qqvalue pvalue, method(holm) qvalue(q)

*Output qvalues to matrices we can put into the main excel file
foreach outcome in $continvars $binvars  {
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		sum q if outcome=="`outcome'" & comp=="`comp'"
		mat `outcome'`comp'adjp=r(mean)
	}
}

restore 



/**************************************************
 Output results to excel 
***************************************************/


*First, output p-values to excel
capture rm "$Results/PerceptReactADEs.xlsx"
putexcel set "$Results/PerceptReactADEs.xlsx", replace

putexcel B1=("Green"), bold
putexcel B2=("ADE"), bold
putexcel C2=("95% CI LL"), bold
putexcel D2=("95% CI UL"), bold
putexcel E2=("adj p-value"), bold

putexcel F1=("TLL"), bold
putexcel F2=("ADE"), bold
putexcel G2=("95% CI LL"), bold
putexcel H2=("95% CI UL"), bold
putexcel I2=("adj p-value"), bold

putexcel J1=("PACE"), bold
putexcel J2=("ADE"), bold
putexcel K2=("95% CI LL"), bold
putexcel L2=("95% CI UL"), bold
putexcel M2=("adj p-value"), bold

putexcel N1=("Warn"), bold
putexcel N2=("ADE"), bold
putexcel O2=("95% CI LL"), bold
putexcel P2=("95% CI UL"), bold
putexcel Q2=("adj p-value"), bold

putexcel R1=("FOPL comparisons to one another, corrected p-values"), bold
putexcel R2=("Green vs. TLL"), bold
putexcel S2=("Green vs. PACE"), bold
putexcel T2=("Green vs. Warn"), bold
putexcel U2=("TLL vs. PACE"), bold
putexcel V2=("TLL vs. Warn"), bold
putexcel W2=("Pace vs. Warn"), bold


putexcel A3=("Label reactions"), bold

local row=4
foreach outcome in $labelrxnsordered  {
	putexcel A`row'=("`outcome'"), bold
	
	local x=2 //start in B 
	local y=3
	local z=4
	local a=5
	
	foreach cond in 2 3 4 5  {
		local col1: word `x' of `c(alpha)' 
		local col2: word `y' of `c(alpha)'
		local col3: word `z' of `c(alpha)'
		local col4: word `a' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.00) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.00;(-0.00")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.00);, -0.00) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001"
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'<0.001 {
			putexcel `col4'`row'=("<.001")
		}
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'==1.0 {
			putexcel `col4'`row'=(">.99")
		}
		
		*Format sig digits of p-values per journal policy
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'>=0.01 & `adjp'<1.0 {
			putexcel `col4'`row', overwritefmt nformat(.00)
		}
		
		
		local x=`x'+4
		local y=`y'+4
		local z=`z'+4
		local a=`a'+4
	}
	
	*Comparisons
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}

putexcel A`row'=("Healthfulness perceptions"), bold
local ++row
foreach outcome in $healthfulordered {
	putexcel A`row'=("`outcome'"), bold
	
	local x=2 //start in B 
	local y=3
	local z=4
	local a=5
	
	foreach cond in 2 3 4 5  {
		local col1: word `x' of `c(alpha)' 
		local col2: word `y' of `c(alpha)'
		local col3: word `z' of `c(alpha)'
		local col4: word `a' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.00) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.00;(-0.00")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.00);, -0.00) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001" and 1.00 with ">0.99"
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'<0.001 {
			putexcel `col4'`row'=("<.001")
		}
		if `adjp'==1.0 {
			putexcel `col4'`row'=(">.99")
		}
		
		*Format sig digits of p-values per journal policy
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'>=0.01 & `adjp'<1.0 {
			putexcel `col4'`row', overwritefmt nformat(.00)
		}
		
		local x=`x'+4
		local y=`y'+4
		local z=`z'+4
		local a=`a'+4
	}
	
	*Comparisons
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}


putexcel A`row'=("Stigma"), bold
local ++row
foreach outcome in $stigmaordered {
	putexcel A`row'=("`outcome'"), bold
	
	local x=2 //start in B 
	local y=3
	local z=4
	local a=5
	
	foreach cond in 2 3 4 5  {
		local col1: word `x' of `c(alpha)' 
		local col2: word `y' of `c(alpha)'
		local col3: word `z' of `c(alpha)'
		local col4: word `a' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.00) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.00;(-0.00")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.00);, -0.00) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001"
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'<0.001 {
			putexcel `col4'`row'=("<.001")
		}
		if `adjp'==1.0 {
			putexcel `col4'`row'=(">.99")
		}

		*Format sig digits of p-values per journal policy
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'>=0.01 & `adjp'<1.0 {
			putexcel `col4'`row', overwritefmt nformat(.00)
		}
				
		local x=`x'+4
		local y=`y'+4
		local z=`z'+4
		local a=`a'+4
	}
	
	*Comparisons
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}


putexcel A`row'=("Positive label perceptions"), bold
local ++row
foreach outcome in $poslabelordered {
	putexcel A`row'=("`outcome'"), bold
	
	local x=2 //start in B 
	local y=3
	local z=4
	local a=5
	
	foreach cond in 2 3 4 5  {
		local col1: word `x' of `c(alpha)' 
		local col2: word `y' of `c(alpha)'
		local col3: word `z' of `c(alpha)'
		local col4: word `a' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.00) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.00;(-0.00")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.00);, -0.00) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001"
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'<0.001 {
			putexcel `col4'`row'=("<.001")
		}
		if `adjp'==1.0 {
			putexcel `col4'`row'=(">.99")
		}
		
		*Format sig digits of p-values per journal policy
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'>=0.01 & `adjp'<1.0 {
			putexcel `col4'`row', overwritefmt nformat(.00)
		}		
		
		local x=`x'+4
		local y=`y'+4
		local z=`z'+4
		local a=`a'+4
	}
	
	*Comparisons
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}


  

*Clean up formatting

*Change logit results to %s 
foreach letter in B F J N {
foreach cells in "`letter'4" "`letter'10:`letter'11" "`letter'27"  {
	putexcel `cells', overwritefmt nformat(0.0%)
}	
}

foreach letter in C G K O {
foreach cells in "`letter'4" "`letter'10:`letter'11" "`letter'27" {
	putexcel `cells',  nformat("(0.0%;(-0.0%")
}
}

foreach letter in D H L P {
	foreach cells in "`letter'4" "`letter'10:`letter'11" "`letter'27" {
	putexcel `cells', nformat(", 0.0%);, -0.0%) ")
}
}
