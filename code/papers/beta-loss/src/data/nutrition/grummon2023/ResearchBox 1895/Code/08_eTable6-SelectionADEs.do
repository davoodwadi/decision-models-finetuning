*********Impact of FOPLs on Selection Outcomes*********

/**************
This do file conducts analyses of FOPL impact on selection outcomes for beverages and for snacks. We also correct p-values for multiple hypothesis testing and output results to the appropriate tables in Excel. Perceptions and label reactions analsyes are in a separate file. 

Specifically, this do file:
-Examines how FOPL arms impact beverage and snack selection outcomes, including estimating average differential effects (ADEs) for each intervention arm vs. control and calculating p-values for significance of these differences vs. control. This includes checking assumptions of ologit and mlogit models. 
-Compares the ADEs to one another, including p-value for difference
-Adjusts p-values for multiple testing using Bonferroni Holm method and the qqvalue package
-Outputs these ADEs  and p-values to Excel. These form basis for Supplemental Table 5 and for any discussion of which labeling arms differed from one another in their impacts on selection outcomes, including superscripts in Figure 3.

**************/

*Bring in the cleaned data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear


/**************************************************************
Conduct analyses for beverages
***************************************************************/


*****
*Primary outcome - calories selected
*****

*Run the TPM and store the results
foreach dv in item_cal_bev { 
tpm `dv' i.foplcond, first(logit) second(regress) robust
	margins, dydx(foplcond) post
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
	tpm `dv' i.foplcond, first(logit) second(regress) robust
		margins, dydx(foplcond) post
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
 
****
*Moderation analyses for primary outcome
****

tpm item_cal_bev i.foplcond##i.loweduc, first(logit) second(regress) robust
testparm i.foplcond#i.loweduc
 
****
*Secondary outcome - likelihood of selecting healthy/unhealthy/moderately healthy
****

/*Try ologit first, then MNL if needed 
ologit RYG_bev_num i.foplcond
	margins, dydx(foplcond)
*	check proportional odds assumption with Brant test
	//see: https://docs.google.com/viewer?url=https%3A%2F%2Fwww3.nd.edu%2F~rwilliam%2Fstats3%2FOlogit01.pdf
	brant, detail
	//violated for TLL - try mlogit instead
	
	*First, check for IIA first
	mlogit RYG_bev_num i.foplcond
	mlogtest, haus //not positive definite, use Small-Hsiao instead
	
	*Must set seed directly before running Small-Hsiao 
	set seed 8675309
	mlogit RYG_bev_num i.foplcond
	mlogtest, smhsiao 
	//not rejected, okay to use MNL
*/

foreach dv in RYG_bev_num { 
	
	*First, output all margins
	mlogit `dv' i.foplcond, base(1)
	margins, dydx(foplcond) post coefleg
	mat rtable = r(table)
	
	*Next, output margins for each outcome level separately for adjusting p's
		*Green
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(1)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Gps=r(mtest)
		
		*Yellow
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(2)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Yps=r(mtest)
			
		*Red
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(3)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Rps=r(mtest)
	

	local col=4 
	local row=1 
	foreach level in 2 3 4 5  { 

		*Predict G
		mat `dv'G`level'b = rtable[1,`col'] 
		mat `dv'G`level'lb = rtable[5,`col'] 
		mat `dv'G`level'ub = rtable[6,`col'] 	
		mat `dv'G`level'adjp=`dv'Gps[`row',4]
		local ++col
		
		*Predict Y
		mat `dv'Y`level'b = rtable[1,`col'] 
		mat `dv'Y`level'lb = rtable[5,`col'] 
		mat `dv'Y`level'ub = rtable[6,`col'] 	
		mat `dv'Y`level'adjp=`dv'Yps[`row',4]
		local ++col
		
		*Predict R
		mat `dv'R`level'b = rtable[1,`col'] 
		mat `dv'R`level'lb = rtable[5,`col'] 
		mat `dv'R`level'ub = rtable[6,`col'] 	
		mat `dv'R`level'adjp=`dv'Rps[`row',4]
		local ++col

		local ++row
	}
	
	
	*Test ADEs against one another, store p-values for later adjustment 
		foreach outcome in 1 2 3 {
		mlogit `dv' i.foplcond, base(1)
			if "`outcome'"=="1" {
				local outletter = "G"
			}
			
			else if "`outcome'"=="2" {
				local outletter = "Y"
			}
			
			else if "`outcome'"=="3" {
				local outletter = "R"
			}
			
			margins, dydx(foplcond) predict(outcome(`outcome')) post 
			
			*Green vs. others
			forval x=3/5 {
				lincom 2.foplcond-`x'.foplcond
				mat `dv'`outletter'2vs`x'p=r(p)
			}
			
			*TLL vs others
			forval x=4/5 {
				lincom 3.foplcond-`x'.foplcond
				mat `dv'`outletter'3vs`x'p=r(p)
			}
			
			*PACE vs others
			foreach x in 5 {
				lincom 4.foplcond-`x'.foplcond
				mat `dv'`outletter'4vs`x'p=r(p)
			}
		}
}
	
****
*Secondary outcome - sugar
****

foreach dv in item_sug_bev { 
tpm `dv' i.foplcond, first(logit) second(regress) robust
	margins, dydx(foplcond) post
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
	tpm `dv' i.foplcond, first(logit) second(regress) robust
		margins, dydx(foplcond) post
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
	

	
****
*Secondary outcome - sodium
****

*Can use regress - not zero inflated
 
 foreach dv in item_sodium_bev { 
	reg `dv' i.foplcond
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
	reg `dv' i.foplcond	
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
	

*All beverages have 0 satfat so we will not examine satfat as an outcome here





/**************************************************************


Conduct analyses for snacks								*******		


***************************************************************/


*****
*Primary outcome - calories selected
*****

*Can use regress - not zero inflated
 foreach dv in item_cal_snack { 
	reg `dv' i.foplcond
	
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
	reg `dv' i.foplcond

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
	

****
*Moderation analyses for primary outcome
****

reg item_cal_snack i.foplcond##i.loweduc
testparm i.foplcond#i.loweduc
 
****
*Secondary outcome - likelihood of selecting healthy/unhealthy/moderately healthy
****

	/*Run ologit, check prop odds, then run mlogit if needed, after checking IIA 
	ologit RYG_snack_num i.foplcond
		//check proportional odds assumption with Brant test
		//see: https://docs.google.com/viewer?url=https%3A%2F%2Fwww3.nd.edu%2F~rwilliam%2Fstats3%2FOlogit01.pdf
		
		brant, detail
		//violated, use mlogit
		
		*First, check for IIA first
		mlogit RYG_snack_num i.foplcond
		mlogtest, haus //not all positive definite, use Small-Hsiao instead
		
		*Must set seed DIRECLTY before running Small-Hsiao 
		set seed 8675309
		mlogit RYG_snack_num i.foplcond, base(1)
		mlogtest, smhsiao 
		//not rejected - okay to use MNL
	
	*/
		
foreach dv in RYG_snack_num { 
	
*First, output all margins
	mlogit `dv' i.foplcond, base(1)
	margins, dydx(foplcond) post coefleg
	mat rtable = r(table)
	
	*Next, output margins for each outcome level separately for adjusting p's
		*Green
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(1)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Gps=r(mtest)
		
		*Yellow
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(2)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Yps=r(mtest)
			
		*Red
			mlogit `dv' i.foplcond, base(1)
			margins, dydx(foplcond) predict(outcome(3)) post 
			*Adjust pvalues 
			test 2.foplcond 3.foplcond 4.foplcond 5.foplcond, mtest(holm)
			mat `dv'Rps=r(mtest)
	

	local col=4 
	local row=1 
	foreach level in 2 3 4 5  { 

		*Predict G
		mat `dv'G`level'b = rtable[1,`col'] 
		mat `dv'G`level'lb = rtable[5,`col'] 
		mat `dv'G`level'ub = rtable[6,`col'] 	
		mat `dv'G`level'adjp=`dv'Gps[`row',4]
		local ++col
		
		*Predict Y
		mat `dv'Y`level'b = rtable[1,`col'] 
		mat `dv'Y`level'lb = rtable[5,`col'] 
		mat `dv'Y`level'ub = rtable[6,`col'] 	
		mat `dv'Y`level'adjp=`dv'Yps[`row',4]
		local ++col
		
		*Predict R
		mat `dv'R`level'b = rtable[1,`col'] 
		mat `dv'R`level'lb = rtable[5,`col'] 
		mat `dv'R`level'ub = rtable[6,`col'] 	
		mat `dv'R`level'adjp=`dv'Rps[`row',4]
		local ++col

		local ++row
	}
	
	*Test ADEs against one another, store p-values for later adjustment 
		foreach outcome in 1 2 3 {
		mlogit `dv' i.foplcond, base(1)
			if "`outcome'"=="1" {
				local outletter = "G"
			}
			
			else if "`outcome'"=="2" {
				local outletter = "Y"
			}
			
			else if "`outcome'"=="3" {
				local outletter = "R"
			}
			
			margins, dydx(foplcond) predict(outcome(`outcome')) post 
			
			*Green vs. others
			forval x=3/5 {
				lincom 2.foplcond-`x'.foplcond
				mat `dv'`outletter'2vs`x'p=r(p)
			}
			
			*TLL vs others
			forval x=4/5 {
				lincom 3.foplcond-`x'.foplcond
				mat `dv'`outletter'3vs`x'p=r(p)
			}
			
			*PACE vs others
			foreach x in 5 {
				lincom 4.foplcond-`x'.foplcond
				mat `dv'`outletter'4vs`x'p=r(p)
			}
		}
	
}
	
	
****
*Secondary outcome - sugar
****

*Use TPM b/c this outcome is zero-inflated
foreach dv in  item_sug_snack  {
	tpm `dv' i.foplcond, first(logit) second(regress) robust
	margins, dydx(foplcond) post
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
	tpm `dv' i.foplcond, first(logit) second(regress) robust
		margins, dydx(foplcond) post
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
	
****
*Secondary outcomes - sodium and sat fat 
****

*Can use regress - not zero inflated
 foreach dv in item_satfat_snack item_sodium_snack { 
	reg `dv' i.foplcond
	
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
	reg `dv' i.foplcond
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
	

/**************************************************
 Adjust p-values to q-values using Bonferroni-Holm
***************************************************/

*First, output p-values to excel
capture rm "$Results/SelectionsCompareFOPLs_ps.xlsx"
putexcel set "$Results/SelectionsCompareFOPLs_ps.xlsx", replace

putexcel A1=("outcome")
putexcel B1=("comparison")
putexcel C1=("pvalue")

local row=2 
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev RYG_bev_numG  RYG_bev_numY RYG_bev_numR item_cal_snack item_sug_snack item_satfat_snack item_sodium_snack RYG_snack_numG RYG_snack_numY RYG_snack_numR  {
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		putexcel A`row'=("`outcome'")
		putexcel B`row'=("`comp'")
		putexcel C`row'=matrix(`outcome'`comp'p)	
		local ++row
	}
}

preserve
*Now, import the excel file created w/ the pvalues and use these data to calculate q-values using BH
import excel "$Results/SelectionsCompareFOPLs_ps.xlsx", sheet("Sheet1") firstrow clear

*Calc qvalues, adjusting using Bonferroni Holm method 
bysort outcome: qqvalue pvalue, method(holm) qvalue(q)

*Output qvalues to matrices we can put into the main excel file
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev RYG_bev_numG  RYG_bev_numY RYG_bev_numR item_cal_snack item_sug_snack item_satfat_snack item_sodium_snack RYG_snack_numG RYG_snack_numY RYG_snack_numR  {
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
capture rm "$Results/SelectionADEs.xlsx"
putexcel set "$Results/SelectionADEs.xlsx", replace

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

putexcel A3=("Beverage selection"), bold

local row=4
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev RYG_bev_numG RYG_bev_numY RYG_bev_numR {
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
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.0) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.0;(-0.0")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.0);, -0.0) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001" or >0.99 if ==1.00
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
	
	*Comparisons of FOPLs to one another
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}

putexcel A`row'=("Snack selection"), bold
local ++row

foreach outcome in item_cal_snack item_sug_snack item_sodium_snack item_satfat_snack RYG_snack_numG RYG_snack_numY RYG_snack_numR {
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
		
		putexcel `col1'`row'=matrix(`outcome'`cond'b), nformat(0.0) 
		putexcel `col2'`row'=matrix(`outcome'`cond'lb),  nformat("(0.0;(-0.0")
		putexcel `col3'`row'=matrix(`outcome'`cond'ub), nformat(", 0.0);, -0.0) ")
		putexcel `col4'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001" or >0.99 if ==1.00
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
	
	*Comparisons of FOPLs to one another
	local b=18 //column R 
	foreach comp in 2vs3 2vs4 2vs5 3vs4 3vs5 4vs5 {
		local column: word `b' of `c(alpha)'
		putexcel `column'`row'=matrix(`outcome'`comp'adjp)
		local ++b
	}
	
	local ++row 
}


*Clean up formatting as needed
	*Change mlogit results to %s 
	foreach cells in "B7:B9" "B15:B17" "F7:F9" "F15:F17" "J7:J9" "J15:J17" "N7:N9" "N15:N17" {
		putexcel `cells', overwritefmt nformat(0.0%)
	}

	foreach cells in "C7:C9" "C15:C17" "G7:G9" "G15:G17" "K7:K9" "K15:K17" "O7:O9" "O15:O17" {
		putexcel `cells',  nformat("(0.0%;(-0.0%")
	}
	
	foreach cells in "D7:D9" "D15:D17" "H7:H9" "H15:H17" "L7:L9" "L15:L17" "P7:P9" "P15:P17" {
		putexcel `cells', nformat(", 0.0%);, -0.0%) ")
	}
	

