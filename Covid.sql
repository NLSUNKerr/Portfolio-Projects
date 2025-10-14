SELECT *
FROM PortfolioProjects.coviddeaths;

DESCRIBE PortfolioProjects.coviddeaths;

-- Likelyhood of dying of COVID-19 in the UK 
SELECT Location, date, total_cases, total_deaths, ((total_deaths/total_cases)*100) as Death_percentage
FROM PortfolioProjects.coviddeaths
WHERE location like "United Kingdom"
ORDER by date;

-- Looking at Total Cases vs Population

SELECT 
	location, 
	MAX(total_cases) as Highest_Infection_Count, 
	population,
    MAX(((total_cases/population)*100)) as Percentage_Infected
FROM PortfolioProjects.coviddeaths
WHERE continent != "" 
GROUP by location , population
ORDER by population DESC;

-- Breaking things down by continent
SELECT
	location
	continent,
	Max(cast(total_deaths as SIGNED)) as Total_Deaths_Count
FROM PortfolioProjects.coviddeaths
WHERE continent = '' 
GROUP by location
ORDER by Total_Deaths_Count DESC;

-- Showing country with highest death count
SELECT 
	location, 
	MAX(cast(total_deaths as SIGNED)) as Total_Deaths_Count
FROM PortfolioProjects.coviddeaths
WHERE continent != ""
GROUP by location
ORDER by Total_Deaths_Count DESC;

-- Global Numbers

SELECT 
	date, 
    SUM(new_cases) as Daily_New_Cases_Worldwide,
    SUM(new_deaths) as Daily_New_Deaths_Worldwide,
    (SUM(new_deaths)/SUM(new_cases))*100 as Death_Percentage_Worldwide
FROM PortfolioProjects.coviddeaths
WHERE continent != ""
GROUP by Date
ORDER by date;

-- looking at total population vs Vaccinations
SELECT dea.continent, 
	dea.location, 
    dea.date, 
    dea.population, 
    vac.new_vaccinations, 
    SUM(vac.new_vaccinations) OVER (Partition by dea.location ORDER by dea.location, dea.date) as Cumulative_Vaccinations
FROM PortfolioProjects.coviddeaths dea
JOIN PortfolioProjects.covidvacc vac
	ON dea.date = vac.date and dea.location = vac.location
WHERE dea.continent != "" ;


-- Use CTE

WITH PopVsVacc (continent, location, dat, population, new_vaccination, Cumulative_Vaccinations)
AS(
SELECT dea.continent, 
	dea.location, 
    dea.date, 
    dea.population, 
    vac.new_vaccinations, 
    SUM(vac.new_vaccinations) OVER (Partition by dea.location ORDER by dea.location, dea.date) as Cumulative_Vaccinations
FROM PortfolioProjects.coviddeaths dea
JOIN PortfolioProjects.covidvacc vac 
ON dea.date = vac.date and dea.location = vac.location  
WHERE dea.continent != "" )
SElECT *, (Cumulative_Vaccinations / Population)*100 as Percetage_Vaccinated
FROM PopVsVacc;

-- Creating View to store data for visualization

CREATE View PopVsVacc as
SELECT dea.continent, 
	dea.location, 
    dea.date, 
    dea.population, 
    vac.new_vaccinations, 
    SUM(vac.new_vaccinations) OVER (Partition by dea.location ORDER by dea.location, dea.date) as Cumulative_Vaccinations
FROM PortfolioProjects.coviddeaths dea
JOIN PortfolioProjects.covidvacc vac 
ON dea.date = vac.date and dea.location = vac.location  
WHERE dea.continent != "" 