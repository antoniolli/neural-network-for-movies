import csv

class Movies:

    @staticmethod
    def getDirectors():
        directors = []
        with open('movies.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                first = {"director": row['director1'], "total": row['variance']}
                directors.append(first)

            return directors

    @staticmethod
    def getActors():
        actors = []
        with open('movies.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                first = {"actor": row['actor1'], "total": row['variance']}
                second = {"actor": row['actor2'], "total": row['variance']}
                third = {"actor": row['actor3'], "total": row['variance']}

                result = [first, second, third]
                actors.append(result)

            return actors

    @staticmethod
    def getGenres():
        genres = []
        with open('movies.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                first = {"genre1": row['genre1'], "total": row['variance']}
                second = {"genre2": row['genre2'], "total": row['variance']}
                third = {"genre3": row['genre3'], "total": row['variance']}

                result = [first, second, third]
                genres.append(result)

            return genres


