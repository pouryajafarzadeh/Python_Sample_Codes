import requests
url = 'http://localhost:1360/files/'
multiple_files = [('files', ('1.jpeg', open('a.png', 'rb'), 'image/png')),
                      ('files', ('2.jpeg', open('b.png', 'rb'), 'image/png'))]
r = requests.post(url, files=multiple_files)
print (f'The status code is{r.status_code} and the result is{r.json()}')