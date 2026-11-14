cd "/Users/dlk265/Library/CloudStorage/Box-Box/Atkinson Survey Project 2024/SAF Paper/Replication Files/"
use "Goldfarb_etal_SAF_JCP_replication.dta", clear

* Table 2 *
reg choose_flight i.airline_n i.fuel_type_n i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using table2, word dec(3) label replace

* Figure 2: AMCEs Full Sample *
reg choose_flight i.airline_n i.fuel_type_n i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
coefplot, drop(_cons) omitted baselevels xline(0) ///
headings(1.airline_n = "{bf:Airline}" ///  
1.fuel_type_n = "{bf:Fuel Type}" ///            
1.baggage_n = "{bf:Baggage Allowance}" ///
1.legroom_n = "{bf:Legroom}" ///
1.seat_n = "{bf:Seat Type}" ///
1.price_n = "{bf:Cost}" , labsize(vsmall)) graphregion(color(white)) ylab(, labs(vsmall)) ///
xti(" " "Effect on Probability of Choosing Flight", size(small)) ti("", size(small)) level(95) mlcolor(black) mcolor(black) xlabel(-.3 -.25 -.2 -.15 -.1 -.05 0 .05 .1) ciopts(lcolor(black))

* Figure 3: Moderating Effect of Science Knowledge *
reg choose_flight i.airline_n i.fuel_type_n##c.sciknow_score i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(sciknow_score=(0(1)9))
marginsplot, recast(line) recastci(rarea) ti("")  yline(0) yti(" " "Effect of SAF on Probability of Choosing Flight") scale(.9) plotopts(lcolor(black)) ciopts(color(gs5%50))  xti("Science Knowledge") ylabel(-.05 0 .05 .1 .15)

* Figure 4: Moderating Effect of Political Partisanship *
reg choose_flight i.airline_n i.fuel_type_n##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(party3=(1(1)3))
marginsplot, recast(scatter) ti("") horizontal xline(0)   xti(" " "Effect of SAF vs. Conventional Fuel on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ciopts(lcolor(black)) recastci(rspike) xlabel(-.05 0 .05 .1 .15) yscale(range(.5 3.5)) yti("")
reg choose_flight i.airline_n i.fuel_type_n##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(party3=(1(1)3))
margins, dydx(fuel_type_n) at(party3=(1(1)3)) pwcompare

* Figure 5: Moderating Effect of Climate Change Beliefs, and by Partisanship *
gen climate_action = climate_change
recode climate_action (5=1) (4=1) (3=0) (2=0) (1=0)
* All respondents *
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(climate_action=(0(1)1))
mplotoffset, recast(scatter) recastci(rspike) ti("") horizontal  xti(" " "Effect of SAF on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ylabel(1 " Action" 0 "No action") xline(0) ciopts(lcolor(black))  yti("Climate Change Beliefs") xlabel(-.05 0 .05 .1 .15) legend(pos(5) ring(0)) yscale(range(-.5 1.5))
* By Party *
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=(1(1)3))
mplotoffset, recast(scatter) recastci(rspike) ti("") horizontal  xti(" " "Effect of SAF on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ylabel(1 " Action" 0 "No action") xline(0) ciopts(lcolor(black)) plot1opts(msymbol(square)) plot3opts(msymbol(triangle)) yti("Climate Change Beliefs") xlabel(-.15 -.1 -.05 0 .05 .1 .15 .2) legend(pos(5) ring(0)) yscale(range(-.5 1.5))
* Significance Tests Across Partisan Subgroups Referenced in Manuscript Text *
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=(1(1)3)) 
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=(1(1)3)) pwcompare
* Combined Figure: All Respondents + By Party *
gen counter = _n
gen all = .
gen gop = .
gen dem = .
gen ind = .
gen l = .
gen u = .

reg choose_flight i.airline_n i.fuel_type_n##i.climate_action i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(climate_action=(0(1)1))
replace all = r(table)[1,3] if counter == 4
replace l = r(table)[5,3] if counter == 4
replace u = r(table)[6,3] if counter == 4
replace all = r(table)[1,4] if counter == 8
replace l = r(table)[5,4] if counter == 8
replace u = r(table)[6,4] if counter == 8
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=3) 
replace ind = r(table)[1,3] if counter == 3
replace l = r(table)[5,3] if counter == 3
replace u = r(table)[6,3] if counter == 3
replace ind = r(table)[1,4] if counter == 7
replace l = r(table)[5,4] if counter == 7
replace u = r(table)[6,4] if counter == 7
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=2) 
replace dem = r(table)[1,3] if counter == 2
replace l = r(table)[5,3] if counter == 2
replace u = r(table)[6,3] if counter == 2
replace dem = r(table)[1,4] if counter == 6
replace l = r(table)[5,4] if counter == 6
replace u = r(table)[6,4] if counter == 6
margins, dydx(fuel_type_n) at(climate_action=(0(1)1) party3=1) 
replace gop = r(table)[1,3] if counter == 1
replace l = r(table)[5,3] if counter == 1
replace u = r(table)[6,3] if counter == 1
replace gop = r(table)[1,4] if counter == 5
replace l = r(table)[5,4] if counter == 5
replace u = r(table)[6,4] if counter == 5

recode counter (8=2.15) (7=2.05) (6=1.95) (5=1.85) (4=1.15) (3=1.05) (2=.95) (1=.85)
replace counter = . if counter >=3
gen all_l = .
gen all_u = .
gen gop_l = .
gen gop_u = .
gen dem_l = .
gen dem_u = .
gen ind_l = .
gen ind_u = .
replace all_l = l if all !=.
replace all_u = u if all !=.
replace gop_l = l if gop !=.
replace gop_u = u if gop !=.
replace dem_l = l if dem !=.
replace dem_u = u if dem !=.
replace ind_l = l if ind !=.
replace ind_u = u if ind !=.

twoway (scatter  counter all, msymbol(Sh) mcolor(black)) (scatter counter gop, msymbol(diamond) mcolor(maroon)) (scatter counter dem , msymbol(triangle) mcolor(navy)) (scatter counter ind, mcolor(gs8)) (rspike all_l all_u counter, lcolor(black) horizontal) ///
(rspike dem_l dem_u counter, lcolor(navy) horizontal) ///
(rspike gop_l gop_u counter, lcolor(maroon) horizontal) ///
(rspike ind_l ind_u counter, lcolor(gs8) horizontal), scale(.9) yti("Climate Change Beliefs") ylabel(1 "No Action" 2 "Action") yscale(range(.5 2.5)) xline(0) legend(order(1 "All" 2 "Republicans" 3 "Democrats" 4 "Independents") pos(5) ring(0)) xlabel(-.15 -.1 -.05 0 .05 .1 .15 .2) xti(" " "Effect of SAF on Probability of Choosing Flight")

* Figure 6: Moderating Effect of Safety Perceptions *

reg choose_flight i.airline_n i.fuel_type_n##c.saf_safe i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(saf_safe=(1(1)5))
marginsplot, recast(scatter) ti("") horizontal xline(0) yti("SAFs are Safe") ylabel(5 "Strongly agree" 4 "Agree" 3 "Neutral" 2 "Disagree" 1 "Strongly disagree") xti(" " "Effect of SAF vs. Conventional Fuel on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ciopts(lcolor(black)) recastci(rspike) xlabel(-.1 -.05 0 .05 .1 .15) yscale(range(.5 5.5))
 

* SUPPORTING INFORMATION TABLES AND FIGURES *

* SI Table 1: Sample Demos *
preserve
duplicates drop respondent, force
sum black hispanic female college
sum age, detail
tab party
tab party_lean
sum climate_change
tab saf_safe
* SI Figure 1: Distribution of Science Knowledge *
hist sciknow_score, xlabel(0 1 2 3 4 5 6 7 8 9) percent xti("Science Knowledge Score")
restore 

* SI Table 2: With and Without Weights *
reg choose_flight i.airline_n i.fuel_type_n i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_weights, word dec(3) label replace
reg choose_flight i.airline_n i.fuel_type_n i.baggage_n i.legroom_n i.seat_n i.price_n [pw=weight], cluster(respondent)
outreg2 using si_table_weights, word dec(3) label append

* SI Figure 2: Average Component Preferences *
* Analysis conducted/Figure constructed in R using Ganter package *

* SI Table 3: SAF X Science Interaction *  
reg choose_flight i.airline_n i.fuel_type_n##c.sciknow_score i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_sciknow, word dec(3) label  drop(i.airline_n i.baggage_n i.legroom_n i.seat_n i.price_n) replace

* SI Table 4: SAF X Partisanship Interactions *
reg choose_flight i.airline_n i.fuel_type_n##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_party, word dec(3) label  drop(i.airline_n i.baggage_n i.legroom_n i.seat_n i.price_n) replace

* SI Table 5: SAF X Climate Change Interaction *
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_climate, word dec(3) label  drop(i.airline_n i.baggage_n i.legroom_n i.seat_n i.price_n) replace

* SI Table 6: SAF X Climate Change Interaction by Party *
reg choose_flight i.airline_n i.fuel_type_n##i.climate_action##i.party3 i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_party_climate, word dec(3) label  drop(i.airline_n i.baggage_n i.legroom_n i.seat_n i.price_n) replace

* SI Table 7: SAF X SAF Safety Perceptions *
reg choose_flight i.airline_n i.fuel_type_n##c.saf_safe i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
outreg2 using si_table_safety, word dec(3) label  drop(i.airline_n i.baggage_n i.legroom_n i.seat_n i.price_n) replace

* SI Figure 3: SAF AMCE by Familiarity *
reg choose_flight i.airline_n i.fuel_type_n##c.saf_familiar i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(saf_familiar=(1(1)4))
marginsplot, recast(scatter) ti("") horizontal xline(0) yti("Familiarity with SAFs")  xti(" " "Effect of SAF vs. Conventional Fuel on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ciopts(lcolor(black)) recastci(rspike) xlabel(-.1 -.05 0 .05 .1 .15) yscale(range(.5 4.5))

* SI Figure 4: SAF AMCE by Educational Attainment *
reg choose_flight i.airline_n i.fuel_type_n##i.college i.baggage_n i.legroom_n i.seat_n i.price_n, cluster(respondent)
margins, dydx(fuel_type_n) at(college=(0(1)1))
marginsplot, recast(scatter) ti("") horizontal xline(0)   xti(" " "Effect of SAF vs. Conventional Fuel on Probability of Choosing Flight") scale(.9) plotopts(mcolor(black)) ciopts(lcolor(black)) recastci(rspike) xlabel(-.05 0 .05 .1 .15) yscale(range(-.5 1.5)) yti("") ylabel(0 "No College Degree" 1 "College Degree")
margins, dydx(fuel_type_n) at(college=(0(1)1)) pwcompare

* WTP Estimate *
* Chiambaretto approach *
* Take AMCE of maximum price value and divide by dollar difference from baseline to max price *
di .304/75
* This is the decrease in prob. of choosing a flight per dollar spent *
* Divide the AMCE of SAF by the figure above *
di .0690679/(-.3039845 /75)

* This method assumes linearity in the effects of price on choice for the estimation of WTP *

* Alternately, we could re-estimate our model treating price as a continuous attribute instead of as a categorical one. *
* As we saw in Table 1, the effect is roughly linear as price increases in $25 increments from $250 to $325, though the effect gets a bit larger at the upper end of the price scale. * 
gen price_usd = .
replace price_usd = 250 if price_n == 0
replace price_usd = 275 if price_n == 1
replace price_usd = 300 if price_n == 2
replace price_usd = 325 if price_n == 3

reg choose_flight i.airline_n i.fuel_type_n i.baggage_n i.legroom_n i.seat_n price_usd, cluster(respondent)
nlcom (WTP_SAF: _b[2.fuel_type_n] / -_b[price_usd])
