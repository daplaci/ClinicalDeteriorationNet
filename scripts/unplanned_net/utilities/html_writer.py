import numpy as np

class HtmlWriter:
    def __init__(self, filename):
        self.style = """<style type='text/css'>
html {
  font-family: Courier;
}
midnightblue {color : #191970}
navy {color : #000080}
darkblue {color : #00008B}
mediumblue {color : #0000CD}
blue {color : #0000FF}
cornflowerblue {color : #6495ED}
royalblue {color : #4169E1}
dodgerblue {color : #1E90FF}
deepskyblue {color : #00BFFF}
lightskyblue {color : #87CEFA}
black{color:#000000}
lightsalmon {color:	#FFA07A}
salmon {color:	#FA8072}
darksalmon {color:	#E9967A}
lightcoral {color:	#F08080}
indianred {color:	#CD5C5C}
crimson {color:	#DC143C}
firebrick {color:	#B22222}
red {color:	#FF0000}
darkred {color:	#8B0000}
maroon {color:#800000}
</style>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
<p>
"""

        self.colors = [
"midnightblue",
"navy",
"darkblue",
"mediumblue",
"blue",
"cornflowerblue",
"royalblue",
"dodgerblue",
"deepskyblue",
"lightskyblue",
"black",
"lightsalmon",
"salmon",
"darksalmon",
"lightcoral",
"indianred",
"crimson",
"firebrick",
"red",
"darkred",
"maroon"]

        self.colors = np.array(self.colors)
        self.scale = np.linspace(-0.5,0.5, len(self.colors)-1)
        self.f = open('figures/{}.html'.format(filename), 'w')
        self.f.write('<html>')
        self.f.write(self.style)

    def write_html(self, str_, shap_intensity=0, force_black=False):
        if force_black:
            color='black'
        else:
            color = self.colors[sum(shap_intensity>self.scale)]
        self.f.write(f'<{color}> {str_} </{color}>')
        self.newline()
    
    def newline(self):
        self.f.write("</p><p>")
    
    def set_range(self, mean, std):
        min = mean -2*std
        max = mean +2*std
        self.scale = np.linspace(min, max, len(self.colors)-1)

    def __del__(self):
        self.f.write('</p></html>')
        self.f.close()