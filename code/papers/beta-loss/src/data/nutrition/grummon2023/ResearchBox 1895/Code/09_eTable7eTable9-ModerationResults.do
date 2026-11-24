*********Moderation Analsyes*******


/**************
This do file output results to Excel files for all moderation analyses (i.e., by education for primary outcomes and by BMI status for stigma outcomes).

These results are presented in eTable 7 and eTable 9. 
**************/

*Bring in the cleaned data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear

*Examine moderation of label impact on beverage calories by education
tpm item_cal_bev i.foplcond##i.loweduc, first(logit) second(regress) robust
testparm i.foplcond#i.loweduc
mat p_int_bevcal = r(p)
margins, dydx(foplcond) at(loweduc=(1 0)) post //low educ first, then high educ
mat rtable_bevcal=r(table)


*Examine moderation of label impact on snack calories by education
reg item_cal_snack i.foplcond##i.loweduc
testparm i.foplcond#i.loweduc
mat p_int_snackcal = r(p)
margins, dydx(foplcond) at(loweduc=(1 0)) post 
mat rtable_snackcal=r(table)

*Examine moderation of perceived personal stigma, perceived obesity stigma, and disgust by obesity status
foreach outcome in stigma_personal_merged obesitystigma_avg disgust_merged {
	reg `outcome' i.foplcond##i.obesebmi 
	testparm i.foplcond#i.obesebmi
	mat p_int_`outcome' = r(p)
	margins, dydx(foplcond) at(obesebmi=(0 1)) //non-obese BMI first, then obese BMI
	mat rtable_`outcome'=r(table)
	
}

*******
*Output all results to Excel 
*******

****
*Start with primary outcomes table - Calories selected from beverages and snacks
*This is eTable 7.
****

capture rm "$Results/ModerationTables-Calories.xlsx"
putexcel set "$Results/ModerationTables-Calories.xlsx", replace

*Add headers/row labels
putexcel A2=("Interpretative labeling system")

putexcel B1=("Beverage calories")
putexcel B2=("Lower education")
putexcel B3=("ADE")
putexcel C3=("LL")
putexcel D3=("UL")
putexcel E2=("Higher education")
putexcel E3=("ADE")
putexcel F3=("LL")
putexcel G3=("UL")
putexcel H3=("p interaction")

local row=4
foreach fopl in Green TrafficLight PhysicalActivity NutrientWarning {
	putexcel A`row'=("`fopl'")
	local ++row
}

putexcel B8=("Snack calories")
putexcel B9=("Lower education")
putexcel B10=("ADE")
putexcel C10=("LL")
putexcel D10=("UL")
putexcel E9=("Higher education")
putexcel E10=("ADE")
putexcel F10=("LL")
putexcel G10=("UL")
putexcel H10=("p interaction")

local row=11
foreach fopl in Green TrafficLight PhysicalActivity NutrientWarning {
	putexcel A`row'=("`fopl'")
	local ++row
}

*Add results

*Beverage calories
local excelrow = 4
local matcol = 3
forval foplcond=1/4 {
	*loweduc
	putexcel B`excelrow'=rtable_bevcal[1,`matcol'], nformat(#.0)  //b
	putexcel C`excelrow'=rtable_bevcal[5,`matcol'], nformat("(0.0;(-0.0")  //ll
	putexcel D`excelrow'=rtable_bevcal[6,`matcol'], nformat(", 0.0);, -0.0) ") //ul

	local ++matcol
	*higheduc
	putexcel E`excelrow'=rtable_bevcal[1,`matcol'], nformat(#.0)  //b
	putexcel F`excelrow'=rtable_bevcal[5,`matcol'], nformat("(0.0;(-0.0")  //ll
	putexcel G`excelrow'=rtable_bevcal[6,`matcol'], nformat(", 0.0);, -0.0) ") //ul
	local ++matcol
	
	local ++excelrow
}

*add p-value
putexcel H4=mat(p_int_bevcal), nformat(0.00)

*Snack calories
local excelrow = 11
local matcol = 3
forval foplcond=1/4 {
	*loweduc
	putexcel B`excelrow'=rtable_snackcal[1,`matcol'], nformat(#.0) //b
	putexcel C`excelrow'=rtable_snackcal[5,`matcol'], nformat("(0.0;(-0.0")  //ll
	putexcel D`excelrow'=rtable_snackcal[6,`matcol'], nformat(", 0.0);, -0.0) ") //ul
	
	local ++matcol
	*higheduc
	putexcel E`excelrow'=rtable_snackcal[1,`matcol'], nformat(#.0)  //b
	putexcel F`excelrow'=rtable_snackcal[5,`matcol'], nformat("(0.0;(-0.0")  //ll
	putexcel G`excelrow'=rtable_snackcal[6,`matcol'], nformat(", 0.0);, -0.0) ") //ul
	local ++matcol
	
	local ++excelrow
}

*Add p-value
putexcel H11=mat(p_int_snackcal), nformat(0.00)


****
*Now output stigma outcomes table - personal stigma, obesity stigma, disgust outcomes
*This is eTable 9. 
****

capture rm "$Results/ModerationTables-Stigma.xlsx"
putexcel set "$Results/ModerationTables-Stigma.xlsx", replace

*Add headers/row labels
putexcel A2=("Interpretative labeling system")

putexcel B1=("Perceived personal stigmatization")
putexcel B2=("Lower BMI")
putexcel B3=("ADE")
putexcel C3=("LL")
putexcel D3=("UL")
putexcel E2=("Higher BMI")
putexcel E3=("ADE")
putexcel F3=("LL")
putexcel G3=("UL")
putexcel H3=("p interaction")

local row=4
foreach fopl in Green TrafficLight PhysicalActivity NutrientWarning {
	putexcel A`row'=("`fopl'")
	local ++row
}

putexcel B8=("Perceived obesity stigmatization")
putexcel B9=("Lower BMI")
putexcel B10=("ADE")
putexcel C10=("LL")
putexcel D10=("UL")
putexcel E9=("Higher BMI")
putexcel E10=("ADE")
putexcel F10=("LL")
putexcel G10=("UL")
putexcel H10=("p interaction")

local row=11
foreach fopl in Green TrafficLight PhysicalActivity NutrientWarning {
	putexcel A`row'=("`fopl'")
	local ++row
}


putexcel B15=("Disgust toward people with obesity")
putexcel B16=("Lower BMI")
putexcel B17=("ADE")
putexcel C17=("LL")
putexcel D17=("UL")
putexcel E16=("Higher BMI")
putexcel E17=("ADE")
putexcel F17=("LL")
putexcel G17=("UL")
putexcel H17=("p interaction")

local row=18
foreach fopl in Green TrafficLight PhysicalActivity NutrientWarning {
	putexcel A`row'=("`fopl'")
	local ++row
}


*Add results

*Perceived personal stigma
local excelrow = 4
local matcol = 3
forval foplcond=1/4 {
	*lowBMI
	putexcel B`excelrow'=rtable_stigma_personal_merged[1,`matcol'], nformat(0.00)  //b
	putexcel C`excelrow'=rtable_stigma_personal_merged[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel D`excelrow'=rtable_stigma_personal_merged[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul

	local ++matcol
	*highBMI
	putexcel E`excelrow'=rtable_stigma_personal_merged[1,`matcol'], nformat(0.00)  //b
	putexcel F`excelrow'=rtable_stigma_personal_merged[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel G`excelrow'=rtable_stigma_personal_merged[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul
	local ++matcol
	
	local ++excelrow
}

*add p-value
putexcel H4=mat(p_int_stigma_personal_merged), nformat(0.00)


*Perceived obesity stigma
local excelrow = 11
local matcol = 3
forval foplcond=1/4 {
	*lowbmi
	putexcel B`excelrow'=rtable_obesitystigma_avg[1,`matcol'], nformat(0.00) //b
	putexcel C`excelrow'=rtable_obesitystigma_avg[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel D`excelrow'=rtable_obesitystigma_avg[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul
	
	local ++matcol
	*highbmi
	putexcel E`excelrow'=rtable_obesitystigma_avg[1,`matcol'], nformat(0.00)  //b
	putexcel F`excelrow'=rtable_obesitystigma_avg[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel G`excelrow'=rtable_obesitystigma_avg[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul
	local ++matcol
	
	local ++excelrow
}

*Add p-value
putexcel H11=mat(p_int_obesitystigma_avg), nformat(0.00)



*Disgust toward people with obesity
local excelrow = 18
local matcol = 3
forval foplcond=1/4 {
	*lowbmi
	putexcel B`excelrow'=rtable_disgust_merged[1,`matcol'], nformat(0.00) //b
	putexcel C`excelrow'=rtable_disgust_merged[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel D`excelrow'=rtable_disgust_merged[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul
	
	local ++matcol
	*highbmi
	putexcel E`excelrow'=rtable_disgust_merged[1,`matcol'], nformat(0.00)  //b
	putexcel F`excelrow'=rtable_disgust_merged[5,`matcol'], nformat("(0.00;(-0.00")  //ll
	putexcel G`excelrow'=rtable_disgust_merged[6,`matcol'], nformat(", 0.00);, -0.00) ") //ul
	local ++matcol
	
	local ++excelrow
}

*Add p-value
putexcel H18=mat(p_int_disgust_merged), nformat(0.00)



