we have #questions_verified.csv that we need to verify for issp2021. 
Our verification is on different aspects.
First, we want to check whether question matches data_label in semantics. if not, it would indicate a different question that was written in error during transcription.

Second, we want to see whether type matches the semantics of the question. we want score to only represent ordinal type questions. 

Next only for score questions, we want to see if reversed is correct. reversed is when the ordinal type question is reversed:
reverse False: strongly disagree to strongly agree
reverse True: strongly agree to strongly disagree.

Also there are cases were the question is not correctly specified. for example there might be ellipses where it is not appropriate in the `question` text. Those need to be rewritten. 

Also, the `options` carry the items of the scale, not the `question`. The option would make the statement or ask the question but should not present the choices to select. If you find these cases let me know. 

Also point out questions that are ambiguous, for example referring to or continuing from other questions.

Can you find all potential issues with our csv?