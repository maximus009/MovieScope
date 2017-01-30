from django import forms

class NameForm(forms.Form):
    link = forms.CharField(label='url', max_length=200)

    response=''
    def getForm(self,request):
        form = QueryForm(request.POST)
        if form.is_valid():
            response=form.cleaned_data['query']
            form.save()
        return form