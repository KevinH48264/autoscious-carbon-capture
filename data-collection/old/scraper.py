from scholarly import scholarly, ProxyGenerator

# Set up a free
# pg = ProxyGenerator()
# success = pg.FreeProxies()
# scholarly.use_proxy(pg)
# print("finished setting up proxy")

# search_query = scholarly.search_pubs('carbon capture')
# print("finished creating search generator")

# pub_result = next(search_query)
# print(pub_result)

# {
#     'author_id': ['4bahYMkAAAAJ', 'ruUKktgAAAAJ', ''],
#     'bib': {
#          'abstract': 'Humans can judge from vision alone whether an object is '
#                      'physically stable or not. Such judgments allow observers '
#                      'to predict the physical behavior of objects, and hence '
#                      'to guide their motor actions. We investigated the visual '
#                      'estimation of physical stability of 3-D objects (shown '
#                      'in stereoscopically viewed rendered scenes) and how it '
#                      'relates to visual estimates of their center of mass '
#                      '(COM). In Experiment 1, observers viewed an object near '
#                      'the edge of a table and adjusted its tilt to the '
#                      'perceived critical angle, ie, the tilt angle at which '
#                      'the object',
#          'author': ['SA Cholewiak', 'RW Fleming', 'M Singh'],
#          'pub_year': '2015',
#          'title': 'Perception of physical stability and center of mass of 3-D '
#                   'objects',
#          'venue': 'Journal of vision'
#     },
#     'citedby_url': '/scholar?cites=15736880631888070187&as_sdt=5,33&sciodt=0,33&hl=en',
#     'eprint_url': 'https://jov.arvojournals.org/article.aspx?articleID=2213254',
#     'filled': False,
#     'gsrank': 1,
#     'num_citations': 23,
#     'pub_url': 'https://jov.arvojournals.org/article.aspx?articleID=2213254',
#     'source': 'PUBLICATION_SEARCH_SNIPPET',
#     'url_add_sclib': '/citations?hl=en&xsrf=&continue=/scholar%3Fq%3DPerception%2Bof%2Bphysical%2Bstability%2Band%2Bcenter%2Bof%2Bmass%2Bof%2B3D%2Bobjects%26hl%3Den%26as_sdt%3D0,33&citilm=1&json=&update_op=library_add&info=K8ZpoI6hZNoJ&ei=kiahX9qWNs60mAHIspTIBA',
#     'url_scholarbib': '/scholar?q=info:K8ZpoI6hZNoJ:scholar.google.com/&output=cite&scirp=0&hl=en'
#  }


from scholarly import scholarly, ProxyGenerator
import csv
import json

count = 10

# Set up a free proxy
# pg = ProxyGenerator()
# success = pg.FreeProxies()
# scholarly.use_proxy(pg)
# print("finished setting up proxy")

# Create a search generator to Google Scholar for carbon capture
search_query_gen = scholarly.search_pubs('carbon capture')
pub_result = json.dumps(next(search_query_gen))
print("finished creating search generator")

# Read existing titles from the CSV file and populate the set
processed_pubs = set()
with open('publications.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        pub = row[0]  # Assuming the title is in the first column
        processed_pubs.add(pub)
print("finished processing existing pubs")

# Open the CSV file in append mode
with open('publications.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    try:
        # while pub_result and count > 0:
        for pub_result in search_query_gen:
            print(pub_result, type(pub_result))
            pub_result = json.dumps(pub_result)
            # print(count, pub_result)

            # Check if the publication is already in the CSV file
            if pub_result not in processed_pubs:
                writer.writerow(pub_result)
                processed_pubs.add(pub_result)

            # pub_result = next(search_query_gen)
            count -= 1
            if count == 0:
                break

    except KeyboardInterrupt:
        print("Program interrupted. Writing to CSV file may be incomplete.")

print("Finished writing to CSV.")

# {
#     'author_id': ['4bahYMkAAAAJ', 'ruUKktgAAAAJ', ''],
#     'bib': {
#          'abstract': 'Humans can judge from vision alone whether an object is '
#                      'physically stable or not. Such judgments allow observers '
#                      'to predict the physical behavior of objects, and hence '
#                      'to guide their motor actions. We investigated the visual '
#                      'estimation of physical stability of 3-D objects (shown '
#                      'in stereoscopically viewed rendered scenes) and how it '
#                      'relates to visual estimates of their center of mass '
#                      '(COM). In Experiment 1, observers viewed an object near '
#                      'the edge of a table and adjusted its tilt to the '
#                      'perceived critical angle, ie, the tilt angle at which '
#                      'the object',
#          'author': ['SA Cholewiak', 'RW Fleming', 'M Singh'],
#          'pub_year': '2015',
#          'title': 'Perception of physical stability and center of mass of 3-D '
#                   'objects',
#          'venue': 'Journal of vision'
#     },
#     'citedby_url': '/scholar?cites=15736880631888070187&as_sdt=5,33&sciodt=0,33&hl=en',
#     'eprint_url': 'https://jov.arvojournals.org/article.aspx?articleID=2213254',
#     'filled': False,
#     'gsrank': 1,
#     'num_citations': 23,
#     'pub_url': 'https://jov.arvojournals.org/article.aspx?articleID=2213254',
#     'source': 'PUBLICATION_SEARCH_SNIPPET',
#     'url_add_sclib': '/citations?hl=en&xsrf=&continue=/scholar%3Fq%3DPerception%2Bof%2Bphysical%2Bstability%2Band%2Bcenter%2Bof%2Bmass%2Bof%2B3D%2Bobjects%26hl%3Den%26as_sdt%3D0,33&citilm=1&json=&update_op=library_add&info=K8ZpoI6hZNoJ&ei=kiahX9qWNs60mAHIspTIBA',
#     'url_scholarbib': '/scholar?q=info:K8ZpoI6hZNoJ:scholar.google.com/&output=cite&scirp=0&hl=en'
#  }
