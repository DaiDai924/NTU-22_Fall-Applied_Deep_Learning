import gdown

url = "https://drive.google.com/file/d/1ebfrM68Ox-igr0c_hKIlIvw2ab8ndZEz/view?usp=sharing"
output = "query2course.pkl"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)