DB:
- 5 folders each containing 100 files
- Each file has 4096 samples on separate lines (one column, no header, no timestamps) (aproximately 23 seconds of EEG data per segment?)
- Folder E (S) is ictal (during seizure)

From the research:

"Sets A and B consisted of segments taken from surface EEG recordings that were carried out on five healthy volunteers using a standardized electrode placement scheme (cf. Fig. 1). Volunteers were relaxed in an awake state with eyes open (A) and eyes closed (B), respectively. Sets C, D, and E originated from our EEG archive of presurgical diagnosis. For the present study EEGs from five patients were selected, all of whom had achieved complete seizure control after resection of one of the hippocampal formations, which was therefore correctly diagnosed to be the epileptogenic zone (cf. Fig. 2). Segments in set D were recorded from within the epileptogenic zone, and those in set C from the hippocampal formation of the opposite hemisphere of the brain. While sets C and D contained only activity measured during seizure free intervals, set E only contained seizure activity. Here segments were selected from all recording sites exhibiting ictal activity."

So: 

| Folder | Description | Label |
| :--- | :--- | :--- |
| A (Z) | Surface EEG, eyes open, healthy | 0 |
| B (O) | Surface EEG, eyes closed, healthy | 0 |
| C (N) | Interictal - seizure free interval | 0 |
| D (F) | Intericta; - seizure free interval | 0 |
| E (S) | Ictal - during seizure | 1 |
