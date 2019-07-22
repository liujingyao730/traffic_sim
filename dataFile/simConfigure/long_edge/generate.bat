@echo off
set  /p length= ‰»Î≥§∂»£∫

sumo -c basic.sumocfg -r hill.xml --fcd-output %length%_1.xml
sumo -c basic.sumocfg -r valley.xml --fcd-output %length%_2.xml
sumo -c basic.sumocfg -r even.xml --fcd-output %length%_3.xml

pause