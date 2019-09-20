@echo off
IF EXIST "data100.csv" (
	ren "data.csv" "data500.csv"
	ren "data100.csv" "data.csv"
) ELSE (
	ren "data.csv" "data100.csv"
	ren "data500.csv" "data.csv"
)