import wx
import pickle
import re
from nltk.util import ngrams
from textblob import TextBlob
from sklearn.metrics import accuracy_score

filename = 'SVM_model.sav'
svclassifier = pickle.load(open(filename, 'rb'))

class MyFrame(wx.Frame):

    def __init__(self):
        super().__init__(parent=None, title='Sarcasm Analyser', size = (400,250))
        panel = wx.Panel(self)
        #self.SetBackgroundColour('red')
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.query_text = wx.TextCtrl(panel)
        self.query_text.Bind(wx.EVT_SET_FOCUS,self.toggle_query1)
        self.query_text.Bind(wx.EVT_KILL_FOCUS,self.toggle_query2)
        my_sizer.Add(self.query_text, 0.5, wx.ALL | wx.EXPAND, 5)
        self.query_text.SetFocus()
        
        my_btn = wx.Button(panel, label='Predict')
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5) 

        #txt1 = "Prediction"
        #self.lbl_answer = wx.StaticText(panel,-1,style = wx.ALIGN_LEFT)
        #font = wx.Font(11, wx.ROMAN, wx.NORMAL, wx.NORMAL) 
        #self.lbl_answer.SetFont(font) 
        #self.lbl_answer.SetLabel(txt1)
        #my_sizer.Add(self.lbl_answer, 0) 
        self.answer_text = wx.TextCtrl(panel)
        self.answer_text.Bind(wx.EVT_SET_FOCUS,self.toggle_ans1)
        self.answer_text.Bind(wx.EVT_KILL_FOCUS,self.toggle_ans2)
        my_sizer.Add(self.answer_text, 0, wx.ALL | wx.SHAPED, 5)
        self.answer_text.SetFocus()

        self.percent_text = wx.TextCtrl(panel)
        self.percent_text.Bind(wx.EVT_SET_FOCUS,self.toggle_per1)
        self.percent_text.Bind(wx.EVT_KILL_FOCUS,self.toggle_per2)
        my_sizer.Add(self.percent_text, 0, wx.ALL | wx.SHAPED, 5)
        self.percent_text.SetFocus()

        self.accuracy_text = wx.TextCtrl(panel)
        self.accuracy_text.Bind(wx.EVT_SET_FOCUS,self.toggle_acc1)
        self.accuracy_text.Bind(wx.EVT_KILL_FOCUS,self.toggle_acc2)
        my_sizer.Add(self.accuracy_text, 0, wx.ALL | wx.SHAPED, 5)
        self.accuracy_text.SetFocus()

        my_btn.SetFocus()

        panel.SetSizer(my_sizer)        
        self.Show()

    def toggle_query1(self,evt):
        if self.query_text.GetValue() == "Enter your sarcasm":
            self.query_text.SetValue("")
        evt.Skip()

    def toggle_query2(self,evt):
        if self.query_text.GetValue() == "":
            self.query_text.SetValue("Enter your sarcasm")
        evt.Skip()

    def toggle_ans1(self,evt):
        if self.answer_text.GetValue() == "Prediction":
            self.answer_text.SetValue("")
        evt.Skip()

    def toggle_ans2(self,evt):
        if self.answer_text.GetValue() == "":
            self.answer_text.SetValue("Prediction")
        evt.Skip()

    def toggle_per1(self,evt):
        if self.percent_text.GetValue() == "Percentage %":
            self.percent_text.SetValue("")
        evt.Skip()

    def toggle_per2(self,evt):
        if self.percent_text.GetValue() == "":
            self.percent_text.SetValue("Percentage %")
        evt.Skip()

    def toggle_acc1(self,evt):
        if self.accuracy_text.GetValue() == "Wordlist":
            self.accuracy_text.SetValue("")
        evt.Skip()

    def toggle_acc2(self,evt):
        if self.accuracy_text.GetValue() == "":
            self.accuracy_text.SetValue("Wordlist")
        evt.Skip()

    def on_press(self, event):
        analyse = self.query_text.GetValue()
        analyse = analyse.lower()
        analyse = re.sub(r'[^a-zA-Z0-9\s]', ' ', analyse)
        tokens = [token for token in analyse.split(" ") if token != ""]

        unigram_sum = bigram_sum = trigram_sum = total_sum = pos_high = pos_med = pos_low = neg_high = neg_med = neg_low = 0

        output = list(ngrams(tokens, 1))
        for z in range(len(output)):
            blob = TextBlob(str(output[z]))
            unigram_sum += blob.sentiment.polarity

        output = list(ngrams(tokens, 2))
        for z in range(len(output)):
            blob = TextBlob(str(output[z]))
            bigram_sum += blob.sentiment.polarity

        output = list(ngrams(tokens, 3))
        for z in range(len(output)):
            blob = TextBlob(str(output[z]))
            trigram_sum += blob.sentiment.polarity

        total_sum += unigram_sum + bigram_sum + trigram_sum
        if total_sum <= -1:
            pos_low = 1
            neg_high = 1
        elif total_sum >= 0 and total_sum <= 1:
            pos_med = 1
            neg_med = 1
        elif total_sum >= 2:           
            pos_high = 1
            neg_low = 1

        ans = svclassifier.predict([[pos_high, pos_med, pos_low, neg_high, neg_med, neg_low]])
        
        if ans == 1:
            self.answer_text.SetValue("Sarcastic")
        else:
            self.answer_text.SetValue("Non-Sarcastic")

        percent = (total_sum/3)
        if percent < 0:
            percent *= -100
        else: 
            percent *= 100
        if percent > 100 or percent < -100:
            percent /= 2
        
        percentage = (format(percent, '.2f') + " %")
        accuracy = analyse
        self.percent_text.SetValue(percentage)
        self.accuracy_text.SetValue(accuracy)

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
