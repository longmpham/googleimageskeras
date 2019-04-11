# this scrapes titles descriptions, and urls of websites, not images...


# from bs4 import BeautifulSoup

# import urllib.request as request


# import urllib
# import requests
# from fake_useragent import UserAgent
# from bs4 import BeautifulSoup
# import re

# query = 'dog'
# query = urllib.parse.quote_plus(query) # Format into URL encoding
# number_result = 5


# ua = UserAgent()

# google_url = 'https://www.google.com/search?q=' + query + '&num=' + str(number_result)
# response = requests.get(google_url, {'User-Agent': ua.random})
# soup = BeautifulSoup(response.text, 'html.parser')

# result_div = soup.find_all('div', attrs = {'class': 'g'})

# links = []
# titles = []
# descriptions = []
# to_remove = []
# clean_links = []
# for r in result_div:
# 	# Checks if each element is present, else, raise exception
#     try:
#         link = r.find('a', href = True)
#         title = r.find('h3', attrs={'class': 'r'}).get_text()
#         description = r.find('span', attrs={'class': 'st'}).get_text()

#         for i, l in enumerate(links):
#             clean = re.search('\/url\?q\=(.*)\&sa',l)
            
#             # Anything that doesn't fit the above pattern will be removed
#             if clean is None:
#                 to_remove.append(i)
#                 continue
#             clean_links.append(clean.group(1))

#         for x in to_remove:
#             del titles[x]
#             del descriptions[x]
        
#         # Check to make sure everything is present before appending
#         if link != '' and title != '' and description != '': 
#             links.append(link['href'])
#             titles.append(title)
#             descriptions.append(description)
#     # Next loop if one element is not present
#     except:
#         continue

# print(clean_links, '\n\n')
# # print(titles)
# # print(descriptions, '\n')