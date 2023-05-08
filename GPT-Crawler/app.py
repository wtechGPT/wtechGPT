from flask import Flask, render_template, request, redirect
from scrape import scrape_login


app = Flask("SuperScrapper")

@app.route("/")
def home():
    scrape_login()

@app.route("/scrape")
def scrape():
    Query = {'searchText':request.args.get('searchText'), 'searchType':request.args.get('searchType')}
     
    if Query:
        get_search_infos(Query)
        
        
    else:
        return redirect("/")
    # return render_template("scrapepage.html", searchingBy=place, checkin = checkin, checkout = checkout, adults = adults)
    return "test"

app.run(port=131, debug=True)
