*Analysis Parent File

*Replace xxx with your Stata username
*Replace yyy with the applicable folder location

***
*Set user file paths
***

if "`c(username)'"=="xxx" {
	global Data "yyy/Data"
	global Results "yyy/Results"
	global Code "yyy/Code"
} 
	
***
*Set global variables to be used elsehwere
***

*Set global for duration considered "too fast" - from soft launch. Should be set as 1/3 of median completion time = 573.5 seconds
di  573.5/3
global toofast 191.2

*Set global list of perception and reaction variables - continuous outcomes
global continvars bev_healthyG bev_healthyY bev_healthyR snack_healthyG snack_healthyY snack_healthyR help_bev_merged help_snack_merged attn_merged elab_merged pme_disc_merged pme_enc_merged affect_avg learn_merged trust react_avg stigma_personal obesitystigma_avg disgust

*Set global list of perception and reaction variables - continuous outcomes
global binvars yesnoticed_cal yesnoticed_trial increase_control yesinfl_bev yesinfl_snack 

*Set global lists of perception and reaction variables - ordered for outputting to tables
global labelrxnsordered yesnoticed_trial attn_merged elab_merged affect_avg pme_disc_merged pme_enc_merged yesinfl_bev yesinfl_snack react_avg 
global poslabelordered learn_merged trust increase_control help_bev_merged help_snack_merged
global stigmaordered stigma_personal obesitystigma_avg disgust
global healthfulordered bev_healthyG  bev_healthyY bev_healthyR snack_healthyG snack_healthyY snack_healthyR

***
*Run do files to reproduce analyses
***

***********
*Data cleaning and preparation
***********

*Run do file to clean and prepare the data
do "$Code/01_DataPrep.do"

***********
*Main Text Tables and Figures
***********

*Runs do files to produce tables and figures in main text, in approximate order in which they appear

*Run do file that produces results shown in Figure 2, the CONSORT flow diagram
do "$Code/02_Figure2-CONSORT.do"

*Run do file that produces results shown in Table 1, sample characteristics
do "$Code/03_Table1-ParticipantCharacteristics.do"

*Run do file that produces results shown in Figure 3, beverage and snack selections 
do "$Code/04_Figure3-Selections.do"

*Run do file that produces results shown in Table 2, means and SDs for label perception and label reactions variables
do "$Code/05_Table2-PerceptionReactionMeans.do"

*Run do file that produces results shown in eTable 4, which compares the study sample to the US population
*Note: commented out because this code requires user first downloading PUMS data [data are public, but too large to upload to ResearchBox]
*do "$Code/06_eTable4-CompareSampleUS.do"

*Run do file that produces results shown in eTable 5, means and SDs of selection outcomes
do "$Code/07_eTable5-SelectionMeansSDs.do"

*Run do file that produces results shown in eTable 6, effects of FOPLs on selection outcomes
do "$Code/08_eTable6-SelectionADEs.do"

*Run do file that produces results shown in eTable 7 and eTable 9 - moderation analyses
do "$Code/09_eTable7eTable9-ModerationResults.do"

*Run do file that produces results shown in eTable 8 - impact of FOPLs on label perception and label reactions
do "$Code/10_eTable8-PerceptionReactionADEs.do"




