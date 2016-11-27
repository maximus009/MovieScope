from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
try:
    from django.utils import simplejson as json
except:
    import simplejson as json

# Create your views here.
def post_list(request):
    return render(request, 'home/index.html', {})

def get_name(request):
    form = QueryForm(request.GET)
    print "Free to take the life"
    search = request.GET.get('url')
    #value=simplejson.dumps(some_data)
    print "Yo Mama",search
    return HttpResponse()

