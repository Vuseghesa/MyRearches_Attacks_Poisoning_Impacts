# MyRearches_Attacks_Poisoning_Impacts
Poisoning attack impacts on Machine Learning models in IoT systems

Analyzing Poisoning Attacks on Machine Learning Systems Derived from IoT Data: A Case Study Using Temperature Data

Objective of our study : To examine the phenomen on of the data poisoning attack and its impact on the security of machine learning methods.

Presentation of the dataset.

We exploit a dataset (Cf. file : Iot_temp.csv ) containing temperature readings from IoT devices installed outside and inside an anonymous room. The device was in the test phase. It was therefore uninstalled or switched off several times throughout the sampling period (from 28-07-2018 to 08-12-2018). As technical details, this dataset has 5 columns whose labels are: id: unique identifiers for each reading, room_id/id: identifier of the room in which the device was installed (inside and/or outside), noted_date: date and time of the reading, temp: temperature readings and out/in: whether the sampling was carried out from a device installed inside or outside the room. In total, this dataset contains 97606 lines (see https://www.openml.org/search?type=data&status=active&id=43351&sort=runs ). However, to meet certain analysis tests, we modified some of the data structures without affecting the content, in order to make the results easier to read.

Code description

In this approach, we used the source codes of the KOHEI-MU researcher (see https://www.kaggle.com/code/koheimuramatsu/iot-temperature-forecasting ), but adapted them to our study. The major modification we have made is the addition of a script that allows the poisoning of the dataset. This enabled us to test three types of attack: the clean-label oisining attack, the label-flipping attack, and the sponge poisoning attack. Ultimately, the study we carried out on this dataset containing temperature readings enabled us to analyze and understand the impact of poisoning attacks on the security of machine learning methods. It has also enabled us to open up our perspective towards the development of defense techniques and countermeasures to mitigate the various risks associated with these attacks, which have become recurrent in recent times.
