
library(spatstat)
data(package="spatstat.data")
demo(data)


name_ls = list('anemones', 'betacells', 'bronzefilter', 
             'clmfires', 'finpines', 'longleaf',
            'nbfires', 'shapley', 'spruces', 'waka')
obj_ls = list(anemones, betacells, bronzefilter, 
             clmfires, finpines, longleaf,
            nbfires, shapley, spruces, waka)

for (val in seq(1, 12, by=1))
    {
    data(obj_ls[val])
    write.table(obj_ls[val], file = paste(name_ls[val] , ".csv", sep =''),
                sep = ",", row.names = F)
    }
    