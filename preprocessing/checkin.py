import time
import pandas as pd

def preprocess_totalcheckin(path, save_path, threshold=3):
    if path.endswith('.txt'):
        check_in = pd.read_csv(path, header=None, sep='\t')
        check_in.rename(columns={0:'userid', 1:'time', 2:'lat', 3:'lon', 4:'placeid'}, inplace=True)
    elif path.endswith('.csv'):
        check_in = pd.read_csv(path)
        check_in['time'] = check_in['datetime']
    group = check_in.groupby('placeid').count()
    group.reset_index(inplace=True, drop=False)

    place_id_set = set(list(group[group['userid']<=threshold]['placeid']))
    place_id_filter = check_in['placeid'].map(lambda x: True if x not in place_id_set else False)

    print(f'The number of original trajectory sequences: {len(check_in["userid"].unique())}')
    check_in = check_in[place_id_filter]
    check_in.dropna(axis=0, inplace=True)
    print(f'The number of trajectory sequences after removing tf<={threshold} sparse places: {len(check_in["userid"].unique())}')
    check_in.to_csv(save_path, index=False)
    return check_in


class checkin_preprocess():
    def __init__(self, check_in_path, save_path):
        t0 = time.time()
        print(f'constructing dataset {check_in_path}...')
        if type(check_in_path)==str:
            if check_in_path.endswith('txt'):
                df = pd.read_csv(check_in_path, header=None, sep='\t')
                df = df.rename(columns={0:'userid', 1:'time', 2:'lat', 3:'lon', 4:'placeid'})
            else:
                df = pd.read_csv(check_in_path)
        df = self.temporal_modelling(df)
        df = self.spatial_modelling(df)

        print(df.info())
        print(df.head())
        self.df = df
        print('seq length distribution:')
        print(df.groupby('userid').agg("count")['time'].describe())
        df.to_csv(save_path, index=False)
        t1 = time.time()
        print(f'preprocess complete, {t1-t0} seconds')

    def temporal_modelling(self, df):
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['weekday'] = df['time'].dt.day_of_week
        df['month'] = df['time'].dt.month
        df['week_of_year'] = df['time'].dt.weekofyear
        df['first_day_of_month'] = df['time'].map(lambda x: x.replace(day=1))
        df['week_of_month'] = df['week_of_year']-df['first_day_of_month'].dt.weekofyear
        df['day_of_year'] = df['time'].dt.dayofyear
        df['day_of_month'] = df['time'].dt.day
        df['season'] = df['time'].dt.quarter
        df['timestamp'] = df['time'].apply(lambda x: int(x.value/10**9))
        df = df.sort_values('timestamp')
        print(f'temporal modelling complete')
        return df

    def spatial_modelling(self, df):
        min_lat = df['lat'].min()
        max_lat = df['lat'].max()
        min_lon = df['lon'].min()
        max_lon = df['lon'].max()
        df['normalize_lat'] = (df['lat']-min_lat)/(max_lat-min_lat)
        df['normalize_lon'] = (df['lon']-min_lon)/(max_lon-min_lon)
        print(f'spatial modelling complete')
        return df
