# Baseline модель для работы "Классификация архитектурных стилей"

![test](test.gif)

Целью базового этапа в контексте работы было получить рабочее отправное решение для задачи классификации архитектурных стилей по изображениям зданий и зафиксировать исходный уровень качества, от которого дальше можно отталкиваться при разработке более сложных подходов. Для этого был выбран датасет архитектурных стилей с двадцатью пятью классами, выполнено разбиение на обучающую, валидационную и тестовую выборки, разработан единый пайплайн предобработки изображений и загрузки данных.

На этом основании были реализованы и обучены две базовые модели на основе предобученных сверточных сетей ResNet‑50 и EfficientNet‑B0, где дообучались финальные слои под задачу классификации стилей. Для обеих моделей были получены и проанализированы метрики accuracy, balanced accuracy и macro F1 на тестовой выборке, а также F1 по каждому классу и матрицы ошибок. Результаты показали, что ResNet‑50 даёт около шестидесяти процентов accuracy и немного превосходит EfficientNet‑B0 при сопоставимых настройках обучения, поэтому именно она выбрана в работе как основной бейзлайн, а EfficientNet‑B0 используется для сравнения архитектур и анализа влияния выбора модели на качество распознавания архитектурных стилей.

Датасет: https://drive.google.com/drive/folders/1w5mLc-nUvpBpB547pC2Dl0HnR02oDyqN?usp=sharing (~10,000 изображений).

Остальные файлы на Диске: https://drive.google.com/drive/folders/1jPX0UMXkRx0Voi_GBLp7X6L87tMOB9gv?usp=sharing

Для воспроизводимости .ipynb можно воспользоваться имеющимися файлами в Google Drive. 

## Описание итогов текущего этапа / summary:

В результате обучения ResNet-50 на 15 эпохах с замороженным backbone и дообучением только финального классификатора были получены следующие показатели на тестовой выборке. Общая accuracy составила примерно 0.605, balanced accuracy около 0.607, macro F1 около 0.599, weighted F1 около 0.602. Это означает, что модель правильно классифицирует около 60 процентов изображений, при этом сбалансированность по классам также находится на уровне 60 процентов, что важно для несбалансированного датасета.

Индивидуальные F1 по классам показывают, что модель особенно хорошо справляется с такими стилями, как Ancient Egyptian architecture, Achaemenid architecture, Gothic architecture, Novelty architecture и Russian Revival architecture, где F1 достигает значений 0.84–0.90 и выше. Самыми сложными для ResNet-50 оказались классы Colonial architecture и Edwardian architecture, где F1 заметно ниже и остаётся в диапазоне 0.20–0.30, что говорит о значительных пересечениях этих стилей с другими и недостаточной выразительности глобальных признаков для их уверенного различения.

Для EfficientNet-B0 использовались те же данные, разбиение и параметры обучения. Модель также обучалась 15 эпох с замороженным backbone и дообучением только финального классификатора. Итоговая accuracy на тестовой выборке составила примерно 0.585, balanced accuracy около 0.576, macro F1 около 0.574, weighted F1 около 0.581. Эти значения немного уступают ResNet-50 по всем основным метрикам, хотя различие невелико, порядка 2 процентных пунктов по accuracy и macro F1. При этом EfficientNet-B0 заметно компактнее по числу параметров и объёму файла и обучается быстрее (около 26 минут против примерно часа для ResNet-50 на Colab), что делает её интересной альтернативой при ограниченных ресурсах. По отдельным классам EfficientNet-B0 также показывает высокие F1 для Ancient Egyptian architecture, Achaemenid architecture, Novelty architecture и Romanesque architecture, но ещё хуже, чем ResNet-50, различает Colonial и Edwardian стили.

Сравнение двух моделей показывает, что при одинаковой постановке задачи и одинаковых данных ResNet-50 даёт немного более высокое качество классификации по всем агрегированным метрикам. На уровне отдельных стилей обе модели имеют схожие зоны уверенности и слабости, однако ResNet-50 стабильнее по большинству классов и реже даёт совсем низкие значения F1. EfficientNet-B0 выигрывает по скорости обучения и размеру модели, но по качеству в данном эксперименте остаётся чуть позади. В качестве baseline решения для дальнейшей выпускной работы разумно использовать ResNet-50 как основную модель и EfficientNet-B0 как дополнительную архитектуру для сравнения и анализа влияния выбора backbone на качество классификации архитектурных стилей.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Для ResNet-50: 

Training: 100% 221/221 [30:40<00:00,  8.33s/it, loss=2.1538, acc=0.3710]
Validation: 100% 48/48 [06:25<00:00,  8.02s/it, loss=1.6800, acc=0.4854]
Epoch 1: Train Loss: 2.1538, Train Acc: 0.3710, Val Loss: 1.6800, Val Acc: 0.4854, LR: 0.001000
Training: 100% 221/221 [01:33<00:00,  2.37it/s, loss=1.6156, acc=0.5062]
Validation: 100% 48/48 [00:17<00:00,  2.80it/s, loss=1.5329, acc=0.5186]
Epoch 2: Train Loss: 1.6156, Train Acc: 0.5062, Val Loss: 1.5329, Val Acc: 0.5186, LR: 0.001000
Training: 100% 221/221 [01:33<00:00,  2.36it/s, loss=1.4751, acc=0.5413]
Validation: 100% 48/48 [00:16<00:00,  2.83it/s, loss=1.4704, acc=0.5398]
Epoch 3: Train Loss: 1.4751, Train Acc: 0.5413, Val Loss: 1.4704, Val Acc: 0.5398, LR: 0.001000
Training: 100% 221/221 [01:33<00:00,  2.35it/s, loss=1.4103, acc=0.5637]
Validation: 100% 48/48 [00:17<00:00,  2.71it/s, loss=1.4715, acc=0.5591]
Epoch 4: Train Loss: 1.4103, Train Acc: 0.5637, Val Loss: 1.4715, Val Acc: 0.5591, LR: 0.001000
Training: 100% 221/221 [01:33<00:00,  2.36it/s, loss=1.3719, acc=0.5698]
Validation: 100% 48/48 [00:17<00:00,  2.81it/s, loss=1.5096, acc=0.5139]
Epoch 5: Train Loss: 1.3719, Train Acc: 0.5698, Val Loss: 1.5096, Val Acc: 0.5139, LR: 0.000500
Training: 100% 221/221 [01:32<00:00,  2.40it/s, loss=1.2691, acc=0.6006]
Validation: 100% 48/48 [00:17<00:00,  2.81it/s, loss=1.4142, acc=0.5458]
Epoch 6: Train Loss: 1.2691, Train Acc: 0.6006, Val Loss: 1.4142, Val Acc: 0.5458, LR: 0.000500
Training: 100% 221/221 [01:33<00:00,  2.38it/s, loss=1.2712, acc=0.5975]
Validation: 100% 48/48 [00:16<00:00,  2.85it/s, loss=1.3789, acc=0.5664]
Epoch 7: Train Loss: 1.2712, Train Acc: 0.5975, Val Loss: 1.3789, Val Acc: 0.5664, LR: 0.000500
Training: 100% 221/221 [01:33<00:00,  2.37it/s, loss=1.2473, acc=0.6106]
Validation: 100% 48/48 [00:16<00:00,  2.83it/s, loss=1.3517, acc=0.5697]
Epoch 8: Train Loss: 1.2473, Train Acc: 0.6106, Val Loss: 1.3517, Val Acc: 0.5697, LR: 0.000500
Training: 100% 221/221 [01:32<00:00,  2.38it/s, loss=1.2326, acc=0.6123]
Validation: 100% 48/48 [00:18<00:00,  2.66it/s, loss=1.3326, acc=0.5837]
Epoch 9: Train Loss: 1.2326, Train Acc: 0.6123, Val Loss: 1.3326, Val Acc: 0.5837, LR: 0.000500
Training: 100% 221/221 [01:32<00:00,  2.38it/s, loss=1.2372, acc=0.6121]
Validation: 100% 48/48 [00:17<00:00,  2.76it/s, loss=1.3391, acc=0.5890]
Epoch 10: Train Loss: 1.2372, Train Acc: 0.6121, Val Loss: 1.3391, Val Acc: 0.5890, LR: 0.000250
Training: 100% 221/221 [01:33<00:00,  2.36it/s, loss=1.1810, acc=0.6259]
Validation: 100% 48/48 [00:16<00:00,  2.84it/s, loss=1.3056, acc=0.5976]
Epoch 11: Train Loss: 1.1810, Train Acc: 0.6259, Val Loss: 1.3056, Val Acc: 0.5976, LR: 0.000250
Training: 100% 221/221 [01:33<00:00,  2.36it/s, loss=1.1640, acc=0.6340]
Validation: 100% 48/48 [00:17<00:00,  2.81it/s, loss=1.2892, acc=0.5996]
Epoch 12: Train Loss: 1.1640, Train Acc: 0.6340, Val Loss: 1.2892, Val Acc: 0.5996, LR: 0.000250
Training: 100% 221/221 [01:33<00:00,  2.37it/s, loss=1.1602, acc=0.6338]
Validation: 100% 48/48 [00:17<00:00,  2.69it/s, loss=1.3023, acc=0.5876]
Epoch 13: Train Loss: 1.1602, Train Acc: 0.6338, Val Loss: 1.3023, Val Acc: 0.5876, LR: 0.000250
Training: 100% 221/221 [01:33<00:00,  2.37it/s, loss=1.1596, acc=0.6368]
Validation: 100% 48/48 [00:17<00:00,  2.79it/s, loss=1.2895, acc=0.5923]
Epoch 14: Train Loss: 1.1596, Train Acc: 0.6368, Val Loss: 1.2895, Val Acc: 0.5923, LR: 0.000250
Training: 100% 221/221 [01:33<00:00,  2.36it/s, loss=1.1557, acc=0.6329]
Validation: 100% 48/48 [00:16<00:00,  2.86it/s, loss=1.2909, acc=0.5996]
Epoch 15: Train Loss: 1.1557, Train Acc: 0.6329, Val Loss: 1.2909, Val Acc: 0.5996, LR: 0.000125

Accuracy: 0.6049
Balanced Accuracy: 0.6065
Macro F1-score: 0.5994
Weighted F1-score: 0.6020

F1-score по классам:
 1. Achaemenid architecture                 : 0.8319
 2. American Foursquare architecture        : 0.6129
 3. American craftsman style                : 0.4892
 4. Ancient Egyptian architecture           : 0.8889
 5. Art Deco architecture                   : 0.5333
 6. Art Nouveau architecture                : 0.5897
 7. Baroque architecture                    : 0.5714
 8. Bauhaus architecture                    : 0.4854
 9. Beaux-Arts architecture                 : 0.4098
10. Byzantine architecture                  : 0.6889
11. Chicago school architecture             : 0.5800
12. Colonial architecture                   : 0.2018
13. Deconstructivism                        : 0.6804
14. Edwardian architecture                  : 0.3000
15. Georgian architecture                   : 0.4531
16. Gothic architecture                     : 0.8400
17. Greek Revival architecture              : 0.6490
18. International style                     : 0.5517
19. Novelty architecture                    : 0.8966
20. Palladian architecture                  : 0.4160
21. Postmodern architecture                 : 0.4912
22. Queen Anne architecture                 : 0.7215
23. Romanesque architecture                 : 0.6526
24. Russian Revival architecture            : 0.7719
25. Tudor Revival architecture              : 0.6772

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Для EfficientNet-B0: 

Training: 100% 221/221 [01:30<00:00,  2.44it/s, loss=2.3084, acc=0.3591]
Validation: 100% 48/48 [00:16<00:00,  2.90it/s, loss=1.7740, acc=0.5153]
Epoch 1: Train Loss: 2.3084, Train Acc: 0.3591, Val Loss: 1.7740, Val Acc: 0.5153, LR: 0.001000
Training: 100% 221/221 [01:29<00:00,  2.47it/s, loss=1.7331, acc=0.4952]
Validation: 100% 48/48 [00:16<00:00,  2.93it/s, loss=1.5700, acc=0.5452]
Epoch 2: Train Loss: 1.7331, Train Acc: 0.4952, Val Loss: 1.5700, Val Acc: 0.5452, LR: 0.001000
Training: 100% 221/221 [01:29<00:00,  2.46it/s, loss=1.5832, acc=0.5205]
Validation: 100% 48/48 [00:16<00:00,  2.92it/s, loss=1.4873, acc=0.5551]
Epoch 3: Train Loss: 1.5832, Train Acc: 0.5205, Val Loss: 1.4873, Val Acc: 0.5551, LR: 0.001000
Training: 100% 221/221 [01:29<00:00,  2.47it/s, loss=1.5038, acc=0.5413]
Validation: 100% 48/48 [00:16<00:00,  2.90it/s, loss=1.4479, acc=0.5611]
Epoch 4: Train Loss: 1.5038, Train Acc: 0.5413, Val Loss: 1.4479, Val Acc: 0.5611, LR: 0.001000
Training: 100% 221/221 [01:30<00:00,  2.45it/s, loss=1.4440, acc=0.5579]
Validation: 100% 48/48 [00:16<00:00,  2.87it/s, loss=1.4221, acc=0.5710]
Epoch 5: Train Loss: 1.4440, Train Acc: 0.5579, Val Loss: 1.4221, Val Acc: 0.5710, LR: 0.000500
Training: 100% 221/221 [01:30<00:00,  2.45it/s, loss=1.3878, acc=0.5791]
Validation: 100% 48/48 [00:16<00:00,  2.95it/s, loss=1.3947, acc=0.5684]
Epoch 6: Train Loss: 1.3878, Train Acc: 0.5791, Val Loss: 1.3947, Val Acc: 0.5684, LR: 0.000500
Training: 100% 221/221 [01:30<00:00,  2.44it/s, loss=1.3551, acc=0.5876]
Validation: 100% 48/48 [00:16<00:00,  2.92it/s, loss=1.3841, acc=0.5764]
Epoch 7: Train Loss: 1.3551, Train Acc: 0.5876, Val Loss: 1.3841, Val Acc: 0.5764, LR: 0.000500
Training: 100% 221/221 [01:29<00:00,  2.48it/s, loss=1.3545, acc=0.5818]
Validation: 100% 48/48 [00:16<00:00,  2.83it/s, loss=1.3637, acc=0.5930]
Epoch 8: Train Loss: 1.3545, Train Acc: 0.5818, Val Loss: 1.3637, Val Acc: 0.5930, LR: 0.000500
Training: 100% 221/221 [01:28<00:00,  2.50it/s, loss=1.3396, acc=0.5928]
Validation: 100% 48/48 [00:17<00:00,  2.71it/s, loss=1.3579, acc=0.5803]
Epoch 9: Train Loss: 1.3396, Train Acc: 0.5928, Val Loss: 1.3579, Val Acc: 0.5803, LR: 0.000500
Training: 100% 221/221 [01:28<00:00,  2.50it/s, loss=1.3214, acc=0.5935]
Validation: 100% 48/48 [00:16<00:00,  2.83it/s, loss=1.3540, acc=0.5883]
Epoch 10: Train Loss: 1.3214, Train Acc: 0.5935, Val Loss: 1.3540, Val Acc: 0.5883, LR: 0.000250
Training: 100% 221/221 [01:29<00:00,  2.48it/s, loss=1.2987, acc=0.5922]
Validation: 100% 48/48 [00:16<00:00,  2.93it/s, loss=1.3429, acc=0.5930]
Epoch 11: Train Loss: 1.2987, Train Acc: 0.5922, Val Loss: 1.3429, Val Acc: 0.5930, LR: 0.000250
Training: 100% 221/221 [01:29<00:00,  2.46it/s, loss=1.2897, acc=0.6041]
Validation: 100% 48/48 [00:16<00:00,  2.94it/s, loss=1.3497, acc=0.5870]
Epoch 12: Train Loss: 1.2897, Train Acc: 0.6041, Val Loss: 1.3497, Val Acc: 0.5870, LR: 0.000250
Training: 100% 221/221 [01:29<00:00,  2.46it/s, loss=1.2951, acc=0.6133]
Validation: 100% 48/48 [00:16<00:00,  2.91it/s, loss=1.3418, acc=0.5870]
Epoch 13: Train Loss: 1.2951, Train Acc: 0.6133, Val Loss: 1.3418, Val Acc: 0.5870, LR: 0.000250
Training: 100% 221/221 [01:29<00:00,  2.46it/s, loss=1.2752, acc=0.6048]
Validation: 100% 48/48 [00:16<00:00,  2.91it/s, loss=1.3427, acc=0.5870]
Epoch 14: Train Loss: 1.2752, Train Acc: 0.6048, Val Loss: 1.3427, Val Acc: 0.5870, LR: 0.000250
Training: 100% 221/221 [01:29<00:00,  2.46it/s, loss=1.2859, acc=0.6029]
Validation: 100% 48/48 [00:16<00:00,  2.88it/s, loss=1.3437, acc=0.5883]
Epoch 15: Train Loss: 1.2859, Train Acc: 0.6029, Val Loss: 1.3437, Val Acc: 0.5883, LR: 0.000125

Accuracy: 0.5854
Balanced Accuracy: 0.5756
Macro F1-score: 0.5743
Weighted F1-score: 0.5808

F1-score по классам:
 1. Achaemenid architecture                 : 0.8595
 2. American Foursquare architecture        : 0.4576
 3. American craftsman style                : 0.3551
 4. Ancient Egyptian architecture           : 0.9091
 5. Art Deco architecture                   : 0.5445
 6. Art Nouveau architecture                : 0.6592
 7. Baroque architecture                    : 0.6395
 8. Bauhaus architecture                    : 0.5319
 9. Beaux-Arts architecture                 : 0.4262
10. Byzantine architecture                  : 0.6383
11. Chicago school architecture             : 0.5376
12. Colonial architecture                   : 0.3256
13. Deconstructivism                        : 0.6476
14. Edwardian architecture                  : 0.2899
15. Georgian architecture                   : 0.4200
16. Gothic architecture                     : 0.7961
17. Greek Revival architecture              : 0.6237
18. International style                     : 0.4839
19. Novelty architecture                    : 0.8000
20. Palladian architecture                  : 0.3830
21. Postmodern architecture                 : 0.4468
22. Queen Anne architecture                 : 0.6753
23. Romanesque architecture                 : 0.6596
24. Russian Revival architecture            : 0.7018
25. Tudor Revival architecture              : 0.5455
