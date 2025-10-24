summarize grants_patents RandD_perc_GDP gini_coeff gdp_real_growthrate
emissions_percapita trade_pc_GDP population_mil
gen log_patents_millions = log( grants_patents/ population_mil)
gen GDPpc_Thousands = gdp_percapita/1000
gen interraction_GDPpc_Trade = trade_pc_GDP * GDPpc_Thousand
reg log_patents_millions RandD_perc_GDP gini_coeff gdp_real_growthrate emissions_percapita
GDPpc_Thousand trade_pc_GDP interraction_GDPpc_Trade
rvfplot, yline(0) title("Residuals vs Fitted Values")
//save graph
predict residuals, residuals
kdensity residuals, normal title("Kernel Density Plot of Residuals")
//save graph residuals
swilk residuals
estat imtest
vif
reg log_patents_millions RandD_perc_GDP gini_coeff gdp_real_growthrate emissions_percapita
gdp_percapita trade_pc_GDP
vif