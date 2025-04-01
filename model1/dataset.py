"""
Dataset classes for ASD classification
This module implements custom dataset classes for processing text and audio data for ASD classification.
"""

from torch.utils.data import DataLoader,Dataset
import transformers
import pandas as pd
import torchaudio
import torch

from transformers import  RobertaTokenizerFast
class ASDDataset(Dataset):
    def __init__(self,df:pd.DataFrame,transform=None):
        self.df=df
        self.transform=transformers.ASTFeatureExtractor(sampling_rate=16000,retrun='pt')
        self.sr=16000



    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        wav,sr=torchaudio.load(row.path)
        if wav.shape[0]>1:
            wav=wav.mean(dim=0,keepdim=True)
        wav=torchaudio.functional.resample(wav,sr,self.sr)
        if wav.shape[1]<self.sr*184.32:
            wav=torch.nn.functional.pad(wav,(0,int(self.sr*184.32-wav.shape[1])))
        elif wav.shape[1]>self.sr*184.32:
            wav=wav[:,:self.sr*184.32]
        

        wavs = wav.split(int(16000*10.24),dim=1)
        wav_list = []
        for wav in wavs:
            wav=self.transform(wav.squeeze(0),sampling_rate=self.sr,return_tensors='pt')
            wav_list.append(wav['input_values'])
        wav_list=torch.concat(wav_list)
        return wav_list,torch.tensor(row.language_label,dtype=torch.float32)
    





class ASDWhisperDataset(Dataset):
    """Dataset class for processing audio data using Whisper model."""
    
    def __init__(self, df: pd.DataFrame, label='language_label'):
        self.df = df
        self.processor = transformers.AutoFeatureExtractor.from_pretrained(
            "openai/whisper-base", 
            return_attention_mask=True,
            return_tensors="pt"
        )
        self.sr = 16000
        self.label = label

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = torchaudio.load(row.path)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        wav = torchaudio.functional.resample(wav, sr, self.sr)

        # Pad or truncate to 3 minutes
        if wav.shape[1] < self.sr * 180:
            wav = torch.nn.functional.pad(wav, (0, int(self.sr * 180 - wav.shape[1])))
        elif wav.shape[1] > self.sr * 180:
            wav = wav[:, :self.sr * 180]

        # Split into 30-second segments
        wavs = wav.split(int(16000 * 30), dim=1)
        wav_list = []

        for wav in wavs:
            wav = self.processor(
                wav.squeeze().numpy(), 
                sampling_rate=16000,
                max_length=480000,
                feature_size=3000
            )
            wav_list.append(torch.tensor(wav['input_features'][0]))
        
        wav_list = torch.stack(wav_list, dim=0)
        return wav_list, torch.tensor(row[self.label], dtype=torch.float32)

class ASDWhisperDatasetOverlap(Dataset):
    def __init__(self,df:pd.DataFrame,transform=None,label='language_label'):
        self.df=df
        self.processer = transformers.AutoFeatureExtractor.from_pretrained("openai/whisper-base", return_attention_mask=True,return_tensors="pt")
        self.sr=16000



    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        wav,sr=torchaudio.load(row.path)

        if wav.shape[0]>1:
            wav=wav.mean(dim=0,keepdim=True)

        wav=torchaudio.functional.resample(wav,sr,self.sr)


        if wav.shape[1]<self.sr*180:
            wav=torch.nn.functional.pad(wav,(0,int(self.sr*180-wav.shape[1])))
        elif wav.shape[1]>self.sr*180:
            wav=wav[:,:self.sr*180]
        

        # wavs = wav.split(int(16000*30),dim=1)

        wavs =[ wav[:,i*16000*15:(i+30)*16000*15] for i in range(12)]

        wav_list = []

        for wav in wavs:
            wav = self.processer(wav.squeeze().numpy(), sampling_rate=16000,max_length=480000,feature_size=3000)
            wav_list.append(torch.tensor(wav['input_features'][0]))
        wav_list=torch.stack(wav_list,dim=0)

        return wav_list,torch.tensor(row[self.label],dtype=torch.float32)
    

class ASDWhisperClinicalDataset(Dataset):
    def __init__(self,df:pd.DataFrame,transform=None):
        self.df=df
        self.processer = transformers.AutoFeatureExtractor.from_pretrained("openai/whisper-base", return_attention_mask=True,return_tensors="pt")
        self.sr=16000



    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        wav,sr=torchaudio.load(row.path)

        if wav.shape[0]>1:
            wav=wav.mean(dim=0,keepdim=True)

        wav=torchaudio.functional.resample(wav,sr,self.sr)


        if wav.shape[1]<self.sr*180:
            wav=torch.nn.functional.pad(wav,(0,int(self.sr*180-wav.shape[1])))
        elif wav.shape[1]>self.sr*180:
            wav=wav[:,:self.sr*180]
        
        sex=row.Gender
        age=row.SubjectMonthAge
        wavs = wav.split(int(16000*30),dim=1)
        wav_list = []

        for wav in wavs:
            wav = self.processer(wav.squeeze().numpy(), sampling_rate=16000,max_length=480000,feature_size=3000)
            wav_list.append(torch.tensor(wav['input_features'][0]))
        wav_list=torch.stack(wav_list,dim=0)

        return wav_list,torch.tensor(row.language_label,dtype=torch.float32) ,torch.tensor(sex,dtype=torch.float32),torch.tensor(age,dtype=torch.float32)
    










class CustomRobertaDataset(Dataset):
    def __init__(self,df:pd.DataFrame,stage=1):
        self.df=df
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.processer = transformers.AutoFeatureExtractor.from_pretrained("openai/whisper-base", return_attention_mask=True,return_tensors="pt")
        self.sr=16000
        self.stage=stage


    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        wav,sr=torchaudio.load(row.path)

        if wav.shape[0]>1:
            wav=wav.mean(dim=0,keepdim=True)

        wav=torchaudio.functional.resample(wav,sr,self.sr)


        if wav.shape[1]<self.sr*180:
            wav=torch.nn.functional.pad(wav,(0,int(self.sr*180-wav.shape[1])))
        elif wav.shape[1]>self.sr*180:
            wav=wav[:,:self.sr*180]
        

        wavs = wav.split(int(16000*30),dim=1)
        wav_list = []

        for wav in wavs:
            wav = self.processer(wav.squeeze().numpy(), sampling_rate=16000,max_length=480000,feature_size=3000)
            wav_list.append(torch.tensor(wav['input_features'][0]))
        wav_list=torch.stack(wav_list,dim=0)

        input = self.tokenizer(row.text,return_tensors='pt', truncation=True, padding='max_length', max_length=514)
        input={k: v.squeeze(0) for k,v in input.items()}
        if self.stage==1:
            label = 0 if row.Class_x == 'TD' else 1
        elif self.stage==2:
            label = 1 if row.Class_x == 'ASD' else 0
        subjectid = row.SubjectId

        if row.language_label == 'yes':
            language_label = 1
        else:
            language_label = 0


        input['labels']=label
        input['SubjectId']=subjectid
        input['audio_data']=wav_list
        input['language_label']=torch.tensor(language_label,dtype=torch.long)
        return input
    







class CustomRobertaNoisyAudioDataset(Dataset):
    def __init__(self,df:pd.DataFrame,stage=1):
        self.df=df
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.processer = transformers.AutoFeatureExtractor.from_pretrained("openai/whisper-base", return_attention_mask=True,return_tensors="pt")
        self.sr=16000
        self.stage=stage


    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self,idx:int):
        row=self.df.iloc[idx]
        wav,sr=torchaudio.load(row.path)


        noise_level = 0.25  # Adjust this value to control the noise level
        wav = noise_level * torch.randn_like(wav)
        




        if wav.shape[0]>1:
            wav=wav.mean(dim=0,keepdim=True)

        wav=torchaudio.functional.resample(wav,sr,self.sr)


        if wav.shape[1]<self.sr*180:
            wav=torch.nn.functional.pad(wav,(0,int(self.sr*180-wav.shape[1])))
        elif wav.shape[1]>self.sr*180:
            wav=wav[:,:self.sr*180]
        

        wavs = wav.split(int(16000*30),dim=1)
        wav_list = []

        for wav in wavs:
            wav = self.processer(wav.squeeze().numpy(), sampling_rate=16000,max_length=480000,feature_size=3000)
            wav_list.append(torch.tensor(wav['input_features'][0]))
        wav_list=torch.stack(wav_list,dim=0)

        input = self.tokenizer(row.text,return_tensors='pt', truncation=True, padding='max_length', max_length=514)
        input={k: v.squeeze(0) for k,v in input.items()}
        if self.stage==1:
            label = 0 if row.Class_x == 'TD' else 1
        elif self.stage==2:
            label = 1 if row.Class_x == 'ASD' else 0
        subjectid = row.SubjectId

        if row.language_label == 'yes':
            language_label = 1
        else:
            language_label = 0


        input['labels']=label
        input['SubjectId']=subjectid
        input['audio_data']=wav_list
        input['language_label']=torch.tensor(language_label,dtype=torch.long)
        return input

class MultimodalASDDataset(Dataset):
    """Dataset class for multimodal ASD classification (text + audio)."""
    
    def __init__(self, df: pd.DataFrame, stage=1):
        self.df = df
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.processor = transformers.AutoFeatureExtractor.from_pretrained(
            "openai/whisper-base", 
            return_attention_mask=True,
            return_tensors="pt"
        )
        self.sr = 16000
        self.stage = stage

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Process audio
        wav, sr = torchaudio.load(row.path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = torchaudio.functional.resample(wav, sr, self.sr)

        # Pad or truncate to 3 minutes
        if wav.shape[1] < self.sr * 180:
            wav = torch.nn.functional.pad(wav, (0, int(self.sr * 180 - wav.shape[1])))
        elif wav.shape[1] > self.sr * 180:
            wav = wav[:, :self.sr * 180]

        # Split into 30-second segments
        wavs = wav.split(int(16000 * 30), dim=1)
        wav_list = []

        for wav in wavs:
            wav = self.processor(
                wav.squeeze().numpy(), 
                sampling_rate=16000,
                max_length=480000,
                feature_size=3000
            )
            wav_list.append(torch.tensor(wav['input_features'][0]))
        wav_list = torch.stack(wav_list, dim=0)

        # Process text
        input = self.tokenizer(
            row.text,
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=514
        )
        input = {k: v.squeeze(0) for k, v in input.items()}

        # Prepare labels
        if self.stage == 1:
            label = 0 if row.Class_x == 'TD' else 1
        elif self.stage == 2:
            label = 1 if row.Class_x == 'ASD' else 0

        language_label = 1 if row.language_label == 'yes' else 0

        # Combine all inputs
        input['labels'] = label
        input['SubjectId'] = row.SubjectId
        input['audio_data'] = wav_list
        input['language_label'] = torch.tensor(language_label, dtype=torch.long)
        
        return input