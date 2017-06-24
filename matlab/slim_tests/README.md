# Slimmed-down DCM tests

This folder contains some slimmed down modfied DCM functions. 

We have observed that a substantial portion of many core DCM functions are switches and if statements, with functions doing multiple different things, often recursively, depending on e.g. whether the input is a struct or a cell array. 

For the simple ERP DCM examples, many of these switches are not used, and so they massively overcomplicate the basic task. 

So the slimmed down functions in this folder try to remove all the non-essential, non-core parts of the functions called. 

These are intended as an intermediate step in figuring out the DCM python ports. 

