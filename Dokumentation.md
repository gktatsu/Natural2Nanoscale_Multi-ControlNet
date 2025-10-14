# ControlNet

Das ControlNet dient zur synthetischen Datengenerierung. Dieses ControlNet wurde angepasst, um auf Grundlage jedes
Datensatzes beliebig viele synthetische Daten erzeugen zu können.

------------------------------------------

# Aufbau

Um synthetische Daten zu erzeugen sind die Klassen 'tutorial_train.py', 'tutorial_dataset_ours.py' und 'gradio_seg2image.py'
von Bedeutung. In der Klasse 'tutorial_train.py' werden die Parameter für das Training und die Logger festgelegt. In der Klasse
'tutorial_dataset_ours.py' kann der Pfad zum Datensatz und benötigte Prompts die im Training Verwendung finden angepasst werden.
Die Klasse 'gradio_seg2image.py' implementiert die Inferenz. Um synthetische Daten nach dem Training des Netzwerkes zu erzeugen
müssen die Gewichtungen angepasst werden. Hierfür muss folgende Zeile in 'gradio_seg2image.py' abgeändert werden:

> <span style="color:orange"> model.load_state_dict(load_state_dict('./models/EM_best_results.ckpt', location='cuda')) </span>

------------------------------------------

# Ausführung des Codes (Lokal)

Die Ausführung des Codes (sowohl Training als auch Inferenz) ist lokal extrem schwierig. Da das ControlNet eine Anforderung von 
mindestens 24GB RAM an die GPU stellt wird ein sehr leitungsstarker PC benötigt. Liegt genügend Rechenleistung vor kann die Klasse
'tutorial_train.py' mit den unten aufgeführten Dependencies folgendermaßen ausgeführt werden:
> <span style="color:orange"> python tutorial_train.py </span>

# Ausführung des Codes (Cluster)

Training:
Um das ControlNet zu trainieren wird die Klasse 'tutorial_train.py' ausgeführt.
Bevor das Training durchgeführt wird ist es notwendig in der Klasse 'tutorial_dataset_ours.py' den
Pfad zum Datensatz und gegebenenfalls die Hyperparameter anzupassen. Um den Code
auszuführen kann die Datei 'run.sh' verwendet werden. In 'requirements' sind alle
benötigten dependencies enthalten. Folgender Befehl führt den Code erfolgreich aus:

> <span style="color:orange"> submit ./run.sh --custom hannahkniesel/control:pascal --name NewRun --gpus 3090 </span>

Inferenz:
Um synthetische Daten zu erzeugen wird die Klasse 'gradio_seg2image.py' ausgeführt.
Vor der Ausführung können noch die Anzahl der benötigten Daten und die Prompts angepasst werden.
Der Code wird folgendermaßen ausgeführt:

> <span style="color:orange"> submit ./runGradio.sh --custom hannahkniesel/control:pascal --name NewRun --gpus 3090 </span>

Mit dem verfügbaren DockerContainer werden die zusätzlichen Packages (siehe unten) nicht benötigt. Falls der Container
nicht zur Verfügung steht werden die zusätzlichen Packages ebenfalls benötigt.

------------------------------------------


# Requirements

>- python==<3.8.5>
>- pip==<20.3>
>- cudatoolkit==<11.3>
>- pytorch==<1.12.1>
>- scikit-learn==<1.2.2>
>- tqdm==<4.65.0>
>- numpy==<1.23.1>
>- matplotlib==<3.8.0>
>- torch==<2.1.1>
>- torchvision==<0.16.1>
>- torchaudio==<2.1.1>
>- wandb==<0.16.0>

Zusätzliche Packages:

- gradio==<3.16.2>
- albumentations==<1.3.0>
- opencv-contrib-python==<4.3.0.36>
- imageio==<2.9.0>
- imageio-ffmpeg==<0.4.2>
- pytorch-lightning==<1.5.0>
- omegaconf==<2.1.1>
- test-tube>==<0.7.5>
- streamlit==<1.12.1>
- einops==<0.3.0>
- transformers==<4.19.2>
- webdataset==<0.2.5>
- kornia==<0.6>
- open_clip_torch==<2.0.2>
- invisible-watermark>==<0.1.5>
- streamlit-drawable-canvas==<0.8.0>
- torchmetrics==<0.6.0>
- timm==<0.6.12>
- addict==<2.4.0>
- yapf==<0.32.0>
- prettytable==<3.6.0>
- safetensors==<0.2.7>
- basicsr==<1.4.2>