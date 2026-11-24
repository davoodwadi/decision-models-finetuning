*Participant flow chart*

*Bring in the raw Qualtrics survey data
import delimited "$Data/VLO-Survey-2022-3+FINAL_December+5,+2022_09.36-NoIDs.csv", case(preserve) clear 

*Clicked hyperlink to start survey
count

*Did not consent
count if consent==0

*Too young
count if age<18

*Over quota
count if svy_s =="overquota_hied" | svy_s =="overquota_loed"

*Total did not consent, too young, over quota
count if consent ==0 | age <18 | svy_s =="overquota_hied" | svy_s =="overquota_loed"
drop if consent ==0 | age <18 | svy_s =="overquota_hied" | svy_s =="overquota_loed"
count

*Dropped out 
count if svy_s =="started"
drop if svy_s=="started"
count

tab foplcond 

gen toofast = 1 if Q_TotalD<$toofast
replace toofast=0 if Q_TotalD>=$toofast

count if toofast==1
bysort foplcond: count if toofast==1

count if confirmkeepdata ==0
bysort foplcond: count if confirmkeepdata ==0

gen excludedanalysis=.
replace excludedanalysis=1 if confirmkeepdata==0 | toofast==1 
replace excludedanalysis=0 if confirmkeepdata==1 & toofast==0

drop if toofast==1 | confirmkeepdata==0 
tab foplcond

count
