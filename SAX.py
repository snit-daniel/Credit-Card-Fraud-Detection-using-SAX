import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from saxpy.znorm import znorm
from saxpy.sax import sax_via_window
from saxpy.distance import euclidean
import numpy as npfrom
from saxpy.visit_registry import VisitRegistry
from saxpy.znorm import znorm
from saxpy.hotsax import find_discords_hotsax
from numpy import genfromtxt



# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Create a sidebar with three buttons
page = st.sidebar.radio("Select a page", ["Welcome Page","Import", "Display", "Detection","Contactus"])
# Define the function to register a new user
def register():
    st.write("## Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirmed_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_username and new_password and new_password == confirmed_password:
            with open("users.txt", "a") as f:
                f.write("{}:{}\n".format(new_username, new_password))
            st.success("Successfully registered as {}".format(new_username))
            return True
        else:
            st.error("Please enter a username and matching passwords")
            return False
    return False

# Define the function to sign in a user
def sign_in():
    st.write("## Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        with open("users.txt", "r") as f:
            users = f.read().splitlines()
        for user in users:
            u, p = user.split(":")
            if u == username and p == password:
                st.success("Successfully Logged in as {}".format(username))
                st.session_state.logged_in = True
                return True
        st.error("Invalid username or password")
        return False
    return False

# Check if the user is logged in before allowing access to other pages
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    if not register():
        if not sign_in():
            st.stop()


def welcome_page():
    st.title("Credit Card Fraud Detector App")
    st.write(" This software application uses various techniques and algorithms to identify fraudulent credit card transactions. It is typically used by financial institutions such as banks and credit card companies to protect against fraud and unauthorized transactions. This app typically works by monitoring various aspects of each transaction, including the location of the transaction, the amount of the purchase, and the Day, Time, Distance from home and Time since last transaction of the transactions. It uses of the Symbolic Aggregate Approximation (SAX) technique to detect fraudulent credit card transactions.")

    st.title("What is Credit Card Fraud?")
    st.write("Credit card fraud refers to the unauthorized use of a credit or debit card to make purchases or withdraw cash without the consent of the cardholder. Fraudsters may steal credit card information through various means, such as phishing scams, hacking, skimming, or physical theft of the card. They may use the stolen information to make purchases online or in physical stores, transfer funds, or withdraw cash from ATMs. Credit card fraud can result in financial loss for the cardholder and damage to their credit score. It is important to report any unauthorized transactions immediately to the card issuer to minimize the potential damages.")
    st.video('C:/Users/Welcome/Downloads/How Credit Card Scams Works EMV Card Shimming Bank Fraud and Scams Credit Card Fraud.mp4')


############Define the function to display the import page
def import_page():
    st.title("Import Data Source")


    # Add a file uploader to accept input data
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file was uploaded
    if file is not None:
        # Read the data from the file
        data = pd.read_csv(file)
        st.write("Data preview:")
        st.write(data.head())

        # Store the loaded data in session state
        st.session_state.data = data

################# Define a function to show the display page
def display_page():
    st.title("Display Time series")


    # Check if data has been loaded in session state
    if st.session_state.data is not None:
        data = st.session_state.data


        # Extract the transaction amounts and timestamps
        transaction_amounts = data['Transaction Amount'].values
        timestamps = np.arange(len(transaction_amounts))
        places = data['Place'].values

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for transaction amounts:")
        min_time_t, max_time_t = st.slider("Transaction Amount Time Range", 0, len(transaction_amounts), (0, len(transaction_amounts)), key="transaction_amount_slider")

        # Extract the selected time range for transaction amounts
        selected_transaction_amounts = transaction_amounts[min_time_t:max_time_t]
        selected_timestamps_t = timestamps[min_time_t:max_time_t]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(selected_timestamps_t, selected_transaction_amounts, label="Transaction Amount")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Transaction Amount")
        ax1.legend()
        st.pyplot(fig1)

        # Add a slider to select the time range to be displayed for place
        st.write("Select the time range to be displayed for Place:")
        min_time_p, max_time_p = st.slider("Place Time Range", 0, len(places), (0, len(places)), key="place_slider")

        # Extract the selected time range for place
        selected_places = places[min_time_p:max_time_p]
        selected_timestamps_p = timestamps[min_time_p:max_time_p]

        # Plot the selected time range for place with a larger figure size
        # Define a dictionary mapping numerical values to place names
        place_names = {0: 'Abu Dhabi (0)', 1: 'Dubai (1)', 2: 'Al Ain (2)', 3: 'Sharjah (3)', 4: 'Ajman (4)', 5: 'Fujairah (5)', 6: 'Ras Al Khaimah (6)'}

        # Plot the selected time range for place with a larger figure size
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(selected_timestamps_p, selected_places, label="Place")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Place")

        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax2.set_yticks(list(place_names.keys()))
        ax2.set_yticklabels(list(place_names.values()))

        ax2.legend()
        st.pyplot(fig2)


        # Distance from Home (Miles)

        # Extract the transaction amounts and timestamps

        distance = data['Distance from Home (Miles)'].values
        timestamps = np.arange(len(distance))

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for Distance from Home (Miles):")
        min_distance_d, max_distance_d = st.slider("Distance from Home (Miles) Range", 0, len(distance),
                                           (0, len(distance)), key="Distance from Home (Miles)_slider")

        # Extract the selected time range for transaction amounts
        selected_distance = distance[min_distance_d:max_distance_d]
        selected_distancestamps_d = timestamps[min_distance_d:max_distance_d]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(selected_distancestamps_d, selected_distance, label="Distance from Home (Miles)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Distance from Home (Miles")
        ax3.legend()
        st.pyplot(fig3)

        # Day

        Day = data['Day'].values
        timestamps = np.arange(len(Day))

        # Add a slider to select the time range to be displayed for place
        st.write("Select the time range to be displayed for Days:")
        min_days_d, max_days_d = st.slider("Days Time Range", 0, len(Day), (0, len(Day)), key="Days_slider")

        # Extract the selected time range for place
        selected_days = Day[min_days_d:max_days_d]
        selected_timestamps_d = timestamps[min_days_d:max_days_d]

        # Plot the selected time range for place with a larger figure size
        # Define a dictionary mapping numerical values to place names
        days_names = {0: 'Monday (0)', 1: 'Tuesday (1)', 2: 'Wednesday (2)', 3: 'Thursday (3)', 4: 'Friday (4)',
                      5: 'Saturday(5)', 6: 'Sunday (6)'}

        # Plot the selected time range for place with a larger figure size
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(selected_timestamps_d, selected_days, label="Day")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Days")

        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax4.set_yticks(list(days_names.keys()))
        ax4.set_yticklabels(list(days_names.values()))

        ax4.legend()
        st.pyplot(fig4)


#Time Since Last Transaction (Minutes)

        times = data['Time Since Last Transaction (Minutes)'].values
        timestamps = np.arange(len(times))

        # Add a slider to select the time range to be displayed for transaction amounts
        st.write("Select the time range to be displayed for Time Since Last Transaction (Minutes):")
        min_times_t, max_times_t = st.slider("Time Since Last Transaction (Minutes)Range", 0, len(times),
                                                   (0, len(times)), key="Time Since Last Transaction (Minutes)_slider")

        # Extract the selected time range for transaction amounts
        selected_times = times[min_times_t:max_times_t]
        selected_timestamps_t = timestamps[min_times_t:max_times_t ]

        # Plot the selected time range for transaction amounts with a larger figure size
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(selected_timestamps_t, selected_times, label="Time Since Last Transaction (Minutes)")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Time Since Last Transaction (Minutes)")
        ax5.legend()
        st.pyplot(fig5)
    else:
        st.write("No data has been imported yet.")

# Define the function to display the detection page
def detection_page():
    import streamlit as st
    st.title("Fraud Detection: Displaying discords and Motifs of the dataset")
    st.write("SAX Parameters:")



    # Check if data has been loaded in session state
    if st.session_state.data is not None:
        data = st.session_state.data

        import streamlit as st
        # Create a container for the inputs
        container = st.container()

        # Add inputs to the container
        with container:
            col1, col2, col3, col4 = st.columns(4)
            win_size = col1.number_input('Window size', min_value=0, max_value=200, step=10, value=100)
            num_discords = col2.number_input('Number of discords', min_value=1, max_value=10, step=1, value=2)
            alphabet_size = col3.number_input('Alphabet size', min_value=2, max_value=10, step=1, value=3)
            paa_size = col4.number_input('PAA size', min_value=2, max_value=10, step=1, value=3)
        import numpy as np
        #from saxpy.hotsax import find_discords_hotsax
        from numpy import genfromtxt

        aa = genfromtxt("C:/Users/Admin/Desktop/transaction/Transaction Amount.csv", delimiter=',')
        pp = genfromtxt("C:/Users/Admin/Desktop/transaction/Place.csv", delimiter=',')
        dd = genfromtxt("C:/Users/Admin/Desktop/transaction/Distance from Home (Miles).csv", delimiter=',')
        tt = genfromtxt("C:/Users/Admin/Desktop/transaction/Time Since Last Transaction (Minutes).csv", delimiter=',')
        dday = genfromtxt("C:/Users/Admin/Desktop/transaction/Day.csv", delimiter=',')

        aa = aa[~np.isnan(aa)]
        pp = pp[~np.isnan(pp)]
        dd = dd[~np.isnan(dd)]
        tt = tt[~np.isnan(tt)]
        dday = dday[~np.isnan(dday)]


        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from saxpy.distance import euclidean

        import numpy as np
        from saxpy.visit_registry import VisitRegistry
        # from saxpy.distance import early_abandoned_euclidean
        from saxpy.znorm import znorm

        sax_data = {}  # Define sax_data as a global variable

        def find_discords_hotsax(series, win_size=win_size, num_discords=num_discords, alphabet_size=alphabet_size,
                                 paa_size=paa_size, znorm_threshold=0.01, sax_type='unidim'):
            discords = list()

            global_registry = set()

            # Z-normalized versions for every subsequence.
            znorms = np.array(
                [znorm(series[pos: pos + win_size], znorm_threshold) for pos in range(len(series) - win_size + 1)])

            # SAX words for every subsequence.
            sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                      nr_strategy=None)

            magic_array = list()
            for k, v in sax_data.items():
                magic_array.append((k, len(v)))

            magic_array = sorted(magic_array, key=lambda tup: tup[1])

            while len(discords) < num_discords:

                best_discord = find_best_discord_hotsax(series, win_size, global_registry, sax_data, magic_array,
                                                        znorms)

                if -1 == best_discord[0]:
                    break

                discords.append(best_discord)

                mark_start = max(0, best_discord[0] - win_size + 1)
                mark_end = best_discord[0] + win_size

                for i in range(mark_start, mark_end):
                    global_registry.add(i)

            return discords, sax_data


        def find_best_discord_hotsax(series, win_size, global_registry, sax_data, magic_array, znorms):

            best_so_far_position = -1
            best_so_far_distance = 0.

            distance_calls = 0

            visit_array = np.zeros(len(series), dtype=np.int)

            for entry in magic_array:

                curr_word = entry[0]
                occurrences = sax_data[curr_word]


                for curr_pos in occurrences:

                    if curr_pos in global_registry:
                        continue

                    mark_start = curr_pos - win_size + 1
                    mark_end = curr_pos + win_size
                    visit_set = set(range(mark_start, mark_end))

                    cur_seq = znorms[curr_pos]

                    nn_dist = np.inf
                    do_random_search = True

                    for next_pos in occurrences:

                        if next_pos in visit_set:
                            continue
                        else:
                            visit_set.add(next_pos)


                        dist = euclidean(cur_seq, znorms[next_pos])
                        distance_calls += 1

                        if dist < nn_dist:
                            nn_dist = dist
                        if dist < best_so_far_distance:
                            do_random_search = False
                            break


                    if do_random_search:
                        curr_idx = 0
                        for i in range(0, (len(series) - win_size + 1)):
                            if not (i in visit_set):
                                visit_array[curr_idx] = i
                                curr_idx += 1
                        it_order = np.random.permutation(visit_array[0:curr_idx])
                        curr_idx -= 1

                        while curr_idx >= 0:
                            rand_pos = it_order[curr_idx]
                            curr_idx -= 1

                            dist = euclidean(cur_seq, znorms[rand_pos])
                            distance_calls += 1

                            if dist < nn_dist:
                                nn_dist = dist
                            if dist < best_so_far_distance:
                                nn_dist = dist
                                break

                    if (nn_dist > best_so_far_distance) and (nn_dist < np.inf):
                        best_so_far_distance = nn_dist
                        best_so_far_position = curr_pos

            return best_so_far_position, best_so_far_distance


        ########################################################## motif Transaction Amount#################################################

        import pandas as pd
        import streamlit as st
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from numpy import genfromtxt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Transaction Amount Column")

        series = aa
        st.write("Motif for Transaction Amount")


        # Z-normalized versions for every subsequence.
        znorms = np.array([znorm(series[pos: pos + win_size]) for pos in range(len(series) - win_size + 1)])

        # SAX words for every subsequence.
        sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                  nr_strategy=None)
        print(sax_data)

        sax_listnew = []  # create an empty list to store key-value pairs

        for key in sax_data.keys():  # iterate through the keys in sax_data
            # append the key-value pair as a tuple to sax_listnew,
            # with each value incremented by 100
            sax_listnew.append((key, [value for value in sax_data[key]]))

        # Assign different colors to the motifs
        colors = ['cyan']
        motifs = sax_listnew
        motif_labels = []
        motif_lengths = []
        motif_handles = []

        # Loop through each motif and get the length of its list
        for i, motif in enumerate(motifs):
            motif_key = motif[0]  # Access the key (string) at index 0 of the motif tuple
            motif_len = len(motif[1])  # Get the length of the list at index 1
            motif_lengths.append(motif_len)

            # Add the custom legend handle to the list
            color = colors[i % len(colors)]  # get a color from the list
            motif_handle = Line2D([0], [0], color='w', markerfacecolor=color, marker='s')
            motif_handles.append(motif_handle)

            # Store the label for the motif
            motif_labels.append(motif_key)

        # Create a table to display the motifs and their lengths
        data = {'Motif': motif_labels, 'Frequency': motif_lengths}
        motif_table = pd.DataFrame(data)
        st.table(motif_table)

        # Get the selected motif and plot its corresponding time series data
        selected_motif = st.selectbox('Select a motif to plot:', motif_labels, key='motif_selector')

        # Find the index of the selected motif
        selected_index = motif_labels.index(selected_motif)

        # Clear previous plots
        plt.clf()

        # Plot the selected motif in a separate graph
        plt.figure()

        # Plot the entire time series data
        plt.plot(series)

        # Plot the indexes corresponding to the selected motif
        selected_motif_indexes = motifs[selected_index][1]
        color = colors[selected_index % len(colors)]  # get a color from the list
        for index in selected_motif_indexes:
            window = win_size
            x_start = index
            x_end = index + window
            plt.axvspan(x_start, x_end, color=color, alpha=0.3)

        # Set the title for the selected motif
        plt.title(f"Motif: {selected_motif}")

        # Display the plot within the Streamlit app
        st.pyplot()

        # Display the legend for the selected motif
        plt.legend([motif_handles[selected_index]], [selected_motif])
        plt.show()

    ####################################################### motif end #################################

        ###################################################Transaction Amount#########################################

        discords, sax_data = find_discords_hotsax(aa[0:2000])

        st.write("Discord discovery results for Transaction Amount:")
        for discord in discords:
            st.write(f"Discord position: {discord[0]}, Discord value: {discord[1]}")


        import matplotlib.pyplot as plt
        timestamps = np.arange(len(aa))

        # Add a slider for x-axis range
        st.write("Select the x-axis range:")
        x_min, x_max = st.slider("X-axis Range", 0, len(aa), (0, len(aa)), key="x_axis_range_slider")

        # Extract the selected time range for transaction amounts
        selected_amount = aa[x_min:x_max]
        selected_amountstamps_d = timestamps[x_min:x_max]

        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot the selected time range for transaction amounts with a larger figure size
        ax.plot(selected_amountstamps_d, selected_amount, label="transaction amount")
        ax.set_title('Discords in the Transaction Amount Column', fontsize=36, weight='bold')
        ax.set_xlabel('Index', fontsize=26)
        ax.set_ylabel('Value', fontsize=26)

        for discord in discords:
            discord_start = discord[0]
            discord_end = discord_start + win_size
            discord_values = aa[discord_start:discord_end]

            # Check if the discord falls within the selected x-axis range
            if discord_start >= x_min and discord_end <= x_max:
                ax.plot(np.arange(discord_start, discord_end), discord_values, color='red', label='Discord')
                min_value = min(discord_values)
                middle_index = discord_start + win_size // 2

                # Add SAX word to the discords
                sax_word = None
                for k, v in sax_data.items():
                    if discord_start in v:
                        sax_word = k
                        break

                ax.text(middle_index, min_value, sax_word, ha='center', va='center', color='black', fontsize=17)

        ax.legend()

        # Update the plot with the new x-axis range
        st.pyplot(fig)




        ########################################################## motif Place#################################################

        import pandas as pd
        import streamlit as st
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from numpy import genfromtxt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Place Column")
        series = pp
        import streamlit as st

        st.set_page_config(page_title="Motif for Place", layout="centered")
        st.write("Motif for Place")
        # Z-normalized versions for every subsequence.
        znorms = np.array([znorm(series[pos: pos + win_size]) for pos in range(len(series) - win_size + 1)])

        # SAX words for every subsequence.
        sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                  nr_strategy=None)
        print(sax_data)

        sax_listnew = []  # create an empty list to store key-value pairs

        for key in sax_data.keys():  # iterate through the keys in sax_data
            # append the key-value pair as a tuple to sax_listnew,
            # with each value incremented by 100
            sax_listnew.append((key, [value for value in sax_data[key]]))

        # Assign different colors to the motifs
        colors = ['cyan']
        motifs = sax_listnew
        motif_labels = []
        motif_lengths = []
        motif_handles = []

        # Loop through each motif and get the length of its list
        for i, motif in enumerate(motifs):
            motif_key = motif[0]  # Access the key (string) at index 0 of the motif tuple
            motif_len = len(motif[1])  # Get the length of the list at index 1
            motif_lengths.append(motif_len)

            # Add the custom legend handle to the list
            color = colors[i % len(colors)]  # get a color from the list
            motif_handle = Line2D([0], [0], color='w', markerfacecolor=color, marker='s')
            motif_handles.append(motif_handle)

            # Store the label for the motif
            motif_labels.append(motif_key)

        # Create a table to display the motifs and their lengths
        data = {'Motif': motif_labels, 'Frequency': motif_lengths}
        motif_table = pd.DataFrame(data)
        st.table(motif_table)

        # Get the selected motif and plot its corresponding time series data
        selected_motif = st.selectbox('Select a motif to plot:', motif_labels, key='motif_selector_1')

        # Find the index of the selected motif
        selected_index = motif_labels.index(selected_motif)

        # Clear previous plots
        plt.clf()

        # Plot the selected motif in a separate graph
        plt.figure()

        # Plot the entire time series data
        plt.plot(series)

        # Plot the indexes corresponding to the selected motif
        selected_motif_indexes = motifs[selected_index][1]
        color = colors[selected_index % len(colors)]  # get a color from the list
        for index in selected_motif_indexes:
            window = win_size
            x_start = index
            x_end = index + window
            plt.axvspan(x_start, x_end, color=color, alpha=0.3)

        # Set the title for the selected motif
        plt.title(f"Motif: {selected_motif}")

        # Display the plot within the Streamlit app
        st.pyplot()

        # Display the legend for the selected motif
        plt.legend([motif_handles[selected_index]], [selected_motif])
        plt.show()

        ####################################################### motif end #################################

###################################################Place#########################################

        discords, sax_data = find_discords_hotsax(pp[0:2000])

        st.write("Discord discovery for Place:")
        for discord in discords:
            st.write(f"Discord position: {discord[0]}, Discord value: {discord[1]}")

        timestamps = np.arange(len(pp))

        # Add a slider for x-axis range
        st.write("Select the x-axis range:")
        x_min, x_max = st.slider("X-axis Range", 0, len(pp), (0, len(pp)), key="x_axis_range_slider")

        # Extract the selected time range for transaction amounts
        selected_amount = pp[x_min:x_max]
        selected_amountstamps_d = timestamps[x_min:x_max]

        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot the selected time range for transaction amounts with a larger figure size
        ax.plot(selected_amountstamps_d, selected_amount, label="Place")
        ax.set_title('Discords in the Place Data', fontsize=36, weight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Place")
        place_names = {0: 'Abu Dhabi (0)', 1: 'Dubai (1)', 2: 'Al Ain (2)', 3: 'Sharjah (3)', 4: 'Ajman (4)', 5: 'Fujairah (5)', 6: 'Ras Al Khaimah (6)'}
        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax.set_yticks(list(place_names.keys()))
        ax.set_yticklabels(list(place_names.values()))

        ax.set_xlabel('Index', fontsize=26)
        ax.set_ylabel('Value', fontsize=26)

        for discord in discords:
            discord_start = discord[0]
            discord_end = discord_start + win_size
            discord_values = pp[discord_start:discord_end]

            # Check if the discord falls within the selected x-axis range
            if discord_start >= x_min and discord_end <= x_max:
                ax.plot(np.arange(discord_start, discord_end), discord_values, color='red', label='Discord')
                min_value = min(discord_values)
                middle_index = discord_start + win_size // 2

                # Add SAX word to the discords
                sax_word = None
                for k, v in sax_data.items():
                    if discord_start in v:
                        sax_word = k
                        break

                ax.text(middle_index, min_value, sax_word, ha='center', va='center', color='black', fontsize=17)

        ax.legend()

        # Update the plot with the new x-axis range
        st.pyplot(fig)

        place_names = {0: 'Abu Dhabi (0)', 1: 'Dubai (1)', 2: 'Al Ain (2)', 3: 'Sharjah (3)', 4: 'Ajman (4)', 5: 'Fujairah (5)', 6: 'Ras Al Khaimah (6)'}


        ########################################################## motif Distance from home#################################################

        import pandas as pd
        import streamlit as st
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from numpy import genfromtxt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Distance from Home(Miles) Column")
        series = dd
        st.write("Motif for Distance from Home(miles)")
        # Z-normalized versions for every subsequence.
        znorms = np.array([znorm(series[pos: pos + win_size]) for pos in range(len(series) - win_size + 1)])

        # SAX words for every subsequence.
        sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                  nr_strategy=None)
        print(sax_data)

        sax_listnew = []  # create an empty list to store key-value pairs

        for key in sax_data.keys():  # iterate through the keys in sax_data
            # append the key-value pair as a tuple to sax_listnew,
            # with each value incremented by 100
            sax_listnew.append((key, [value for value in sax_data[key]]))

        # Assign different colors to the motifs
        colors = ['cyan']
        motifs = sax_listnew
        motif_labels = []
        motif_lengths = []
        motif_handles = []

        # Loop through each motif and get the length of its list
        for i, motif in enumerate(motifs):
            motif_key = motif[0]  # Access the key (string) at index 0 of the motif tuple
            motif_len = len(motif[1])  # Get the length of the list at index 1
            motif_lengths.append(motif_len)

            # Add the custom legend handle to the list
            color = colors[i % len(colors)]  # get a color from the list
            motif_handle = Line2D([0], [0], color='w', markerfacecolor=color, marker='s')
            motif_handles.append(motif_handle)

            # Store the label for the motif
            motif_labels.append(motif_key)

        # Create a table to display the motifs and their lengths
        data = {'Motif': motif_labels, 'Frequency': motif_lengths}
        motif_table = pd.DataFrame(data)
        st.table(motif_table)

        # Get the selected motif and plot its corresponding time series data
        selected_motif = st.selectbox('Select a motif to plot:', motif_labels, key='motif_selector_2')

        # Find the index of the selected motif
        selected_index = motif_labels.index(selected_motif)

        # Clear previous plots
        plt.clf()

        # Plot the selected motif in a separate graph
        plt.figure()

        # Plot the entire time series data
        plt.plot(series)

        # Plot the indexes corresponding to the selected motif
        selected_motif_indexes = motifs[selected_index][1]
        color = colors[selected_index % len(colors)]  # get a color from the list
        for index in selected_motif_indexes:
            window = win_size
            x_start = index
            x_end = index + window
            plt.axvspan(x_start, x_end, color=color, alpha=0.3)

        # Set the title for the selected motif
        plt.title(f"Motif: {selected_motif}")

        # Display the plot within the Streamlit app
        st.pyplot()

        # Display the legend for the selected motif
        plt.legend([motif_handles[selected_index]], [selected_motif])
        plt.show()

        ####################################################### motif end #################################
#############################################Distance from Home(Miles)#################################################

        discords, sax_data = find_discords_hotsax(dd[0:2000])

        st.write("Discord discovery for Distance from Home(Miles):")
        for discord in discords:
            st.write(f"Discord position: {discord[0]}, Discord value: {discord[1]}")

        timestamps = np.arange(len(dd))

        # Add a slider for x-axis range
        st.write("Select the x-axis range:")
        x_min, x_max = st.slider("X-axis Range", 0, len(dd), (0, len(dd)), key="x_axis_range_slider")

        # Extract the selected time range for transaction amounts
        selected_amount = dd[x_min:x_max]
        selected_amountstamps_d = timestamps[x_min:x_max]

        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot the selected time range for transaction amounts with a larger figure size
        ax.plot(selected_amountstamps_d, selected_amount, label="transaction amount")
        ax.set_title('Discords in the Distance from Home(Miles) Column', fontsize=36, weight='bold')
        ax.set_xlabel('Index', fontsize=26)
        ax.set_ylabel('Value', fontsize=26)

        for discord in discords:
            discord_start = discord[0]
            discord_end = discord_start + win_size
            discord_values = dd[discord_start:discord_end]

            # Check if the discord falls within the selected x-axis range
            if discord_start >= x_min and discord_end <= x_max:
                ax.plot(np.arange(discord_start, discord_end), discord_values, color='red', label='Discord')
                min_value = min(discord_values)
                middle_index = discord_start + win_size // 2

                # Add SAX word to the discords
                sax_word = None
                for k, v in sax_data.items():
                    if discord_start in v:
                        sax_word = k
                        break

                ax.text(middle_index, min_value, sax_word, ha='center', va='center', color='black', fontsize=17)

        ax.legend()

        # Update the plot with the new x-axis range
        st.pyplot(fig)



        ########################################################## motif Time since last transaction(minute)#################################################

        import pandas as pd
        import streamlit as st
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from numpy import genfromtxt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Time since Last Transaction(Minute) Column")
        series = tt
        st.write("Motif for Time since Last Transaction(Minutes)")
        # Z-normalized versions for every subsequence.
        znorms = np.array([znorm(series[pos: pos + win_size]) for pos in range(len(series) - win_size + 1)])

        # SAX words for every subsequence.
        sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                  nr_strategy=None)
        print(sax_data)

        sax_listnew = []  # create an empty list to store key-value pairs

        for key in sax_data.keys():  # iterate through the keys in sax_data
            # append the key-value pair as a tuple to sax_listnew,
            # with each value incremented by 100
            sax_listnew.append((key, [value for value in sax_data[key]]))

        # Assign different colors to the motifs
        colors = ['cyan']
        motifs = sax_listnew
        motif_labels = []
        motif_lengths = []
        motif_handles = []

        # Loop through each motif and get the length of its list
        for i, motif in enumerate(motifs):
            motif_key = motif[0]  # Access the key (string) at index 0 of the motif tuple
            motif_len = len(motif[1])  # Get the length of the list at index 1
            motif_lengths.append(motif_len)

            # Add the custom legend handle to the list
            color = colors[i % len(colors)]  # get a color from the list
            motif_handle = Line2D([0], [0], color='w', markerfacecolor=color, marker='s')
            motif_handles.append(motif_handle)

            # Store the label for the motif
            motif_labels.append(motif_key)

        # Create a table to display the motifs and their lengths
        data = {'Motif': motif_labels, 'Frequency': motif_lengths}
        motif_table = pd.DataFrame(data)
        st.table(motif_table)

        # Get the selected motif and plot its corresponding time series data
        selected_motif = st.selectbox('Select a motif to plot:', motif_labels, key='motif_selector_3')

        # Find the index of the selected motif
        selected_index = motif_labels.index(selected_motif)

        # Clear previous plots
        plt.clf()

        # Plot the selected motif in a separate graph
        plt.figure()

        # Plot the entire time series data
        plt.plot(series)

        # Plot the indexes corresponding to the selected motif
        selected_motif_indexes = motifs[selected_index][1]
        color = colors[selected_index % len(colors)]  # get a color from the list
        for index in selected_motif_indexes:
            window = win_size
            x_start = index
            x_end = index + window
            plt.axvspan(x_start, x_end, color=color, alpha=0.3)

        # Set the title for the selected motif
        plt.title(f"Motif: {selected_motif}")

        # Display the plot within the Streamlit app
        st.pyplot()

        # Display the legend for the selected motif
        plt.legend([motif_handles[selected_index]], [selected_motif])
        plt.show()

        ####################################################### motif end #################################
##################################################Time Since Last transaction(Minutes)#####################################

        discords, sax_data = find_discords_hotsax(tt[0:2000])

        st.write("Discord discovery for Time Since Last transaction(Minutes):")
        for discord in discords:
            st.write(f"Discord position: {discord[0]}, Discord value: {discord[1]}")



        timestamps = np.arange(len(tt))

        # Add a slider for x-axis range
        st.write("Select the x-axis range:")
        x_min, x_max = st.slider("X-axis Range", 0, len(tt), (0, len(tt)), key="x_axis_range_slider")

        # Extract the selected time range for transaction amounts
        selected_amount = tt[x_min:x_max]
        selected_amountstamps_d = timestamps[x_min:x_max]

        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot the selected time range for transaction amounts with a larger figure size
        ax.plot(selected_amountstamps_d, selected_amount, label="transaction amount")
        ax.set_title('Discords in the Time Since Last transaction(Minutes) Column', fontsize=36, weight='bold')
        ax.set_xlabel('Index', fontsize=26)
        ax.set_ylabel('Value', fontsize=26)

        for discord in discords:
            discord_start = discord[0]
            discord_end = discord_start + win_size
            discord_values = tt[discord_start:discord_end]

            # Check if the discord falls within the selected x-axis range
            if discord_start >= x_min and discord_end <= x_max:
                ax.plot(np.arange(discord_start, discord_end), discord_values, color='red', label='Discord')
                min_value = min(discord_values)
                middle_index = discord_start + win_size // 2

                # Add SAX word to the discords
                sax_word = None
                for k, v in sax_data.items():
                    if discord_start in v:
                        sax_word = k
                        break

                ax.text(middle_index, min_value, sax_word, ha='center', va='center', color='black', fontsize=17)

        ax.legend()

        # Update the plot with the new x-axis range
        st.pyplot(fig)

        ########################################################## Day#################################################

        import pandas as pd
        import streamlit as st
        import numpy as np
        from saxpy.znorm import znorm
        from saxpy.sax import sax_via_window
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        from numpy import genfromtxt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Day Column")
        series = dday
        st.write("Motif for Distance from Home(Miles)")
        # Z-normalized versions for every subsequence.
        znorms = np.array([znorm(series[pos: pos + win_size]) for pos in range(len(series) - win_size + 1)])

        # SAX words for every subsequence.
        sax_data = sax_via_window(series, win_size=win_size, paa_size=paa_size, alphabet_size=alphabet_size,
                                  nr_strategy=None)
        print(sax_data)

        sax_listnew = []  # create an empty list to store key-value pairs

        for key in sax_data.keys():  # iterate through the keys in sax_data
            # append the key-value pair as a tuple to sax_listnew,
            # with each value incremented by 100
            sax_listnew.append((key, [value for value in sax_data[key]]))

        # Assign different colors to the motifs
        colors = ['cyan']
        motifs = sax_listnew
        motif_labels = []
        motif_lengths = []
        motif_handles = []

        # Loop through each motif and get the length of its list
        for i, motif in enumerate(motifs):
            motif_key = motif[0]  # Access the key (string) at index 0 of the motif tuple
            motif_len = len(motif[1])  # Get the length of the list at index 1
            motif_lengths.append(motif_len)

            # Add the custom legend handle to the list
            color = colors[i % len(colors)]  # get a color from the list
            motif_handle = Line2D([0], [0], color='w', markerfacecolor=color, marker='s')
            motif_handles.append(motif_handle)

            # Store the label for the motif
            motif_labels.append(motif_key)

        # Create a table to display the motifs and their lengths
        data = {'Motif': motif_labels, 'Frequency': motif_lengths}
        motif_table = pd.DataFrame(data)
        st.table(motif_table)

        # Get the selected motif and plot its corresponding time series data
        selected_motif = st.selectbox('Select a motif to plot:', motif_labels, key='motif_selector_4')

        # Find the index of the selected motif
        selected_index = motif_labels.index(selected_motif)

        # Clear previous plots
        plt.clf()

        # Plot the selected motif in a separate graph
        plt.figure()

        # Plot the entire time series data
        plt.plot(series)

        # Plot the indexes corresponding to the selected motif
        selected_motif_indexes = motifs[selected_index][1]
        color = colors[selected_index % len(colors)]  # get a color from the list
        for index in selected_motif_indexes:
            window = win_size
            x_start = index
            x_end = index + window
            plt.axvspan(x_start, x_end, color=color, alpha=0.3)

        # Set the title for the selected motif
        plt.title(f"Motif: {selected_motif}")

        # Display the plot within the Streamlit app
        st.pyplot()

        # Display the legend for the selected motif
        plt.legend([motif_handles[selected_index]], [selected_motif])
        plt.show()

        ####################################################### motif end #################################




###################################################Day######################################


        discords, sax_data = find_discords_hotsax(dday[0:2000])

        st.write("Discord discovery for Day:")
        for discord in discords:
            st.write(f"Discord position: {discord[0]}, Discord value: {discord[1]}")


        timestamps = np.arange(len(dday))

        # Add a slider for x-axis range
        st.write("Select the x-axis range:")
        x_min, x_max = st.slider("X-axis Range", 0, len(dday), (0, len(dday)), key="x_axis_range_slider")

        # Extract the selected time range for transaction amounts
        selected_amount = dday[x_min:x_max]
        selected_amountstamps_d = timestamps[x_min:x_max]

        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot the selected time range for transaction amounts with a larger figure size
        ax.plot(selected_amountstamps_d, selected_amount, label="transaction amount")

        days_names = {0: 'Monday (0)', 1: 'Tuesday (1)', 2: 'Wednesday (2)', 3: 'Thursday (3)', 4: 'Friday (4)',
                      5: 'Saturday(5)', 6: 'Sunday (6)'}

        # Set the y-axis tick locations and labels based on the place_names dictionary
        ax.set_yticks(list(days_names.keys()))
        ax.set_yticklabels(list(days_names.values()))

        ax.set_title('Discords in the Day Column', fontsize=36, weight='bold')
        ax.set_xlabel('Index', fontsize=26)
        ax.set_ylabel('Value', fontsize=26)

        for discord in discords:
            discord_start = discord[0]
            discord_end = discord_start + win_size
            discord_values = dday[discord_start:discord_end]

            # Check if the discord falls within the selected x-axis range
            if discord_start >= x_min and discord_end <= x_max:
                ax.plot(np.arange(discord_start, discord_end), discord_values, color='red', label='Discord')
                min_value = min(discord_values)
                middle_index = discord_start + win_size // 2

                # Add SAX word to the discords
                sax_word = None
                for k, v in sax_data.items():
                    if discord_start in v:
                        sax_word = k
                        break

                ax.text(middle_index, min_value, sax_word, ha='center', va='center', color='black', fontsize=17)

        ax.legend()

        # Update the plot with the new x-axis range
        st.pyplot(fig)
        ##

    else:
        st.write("No data has been imported yet.")



def contact_page():



    st.title("Contact The Developer for Any Inqury!")
    st.write("For any question and inquries contact me through snitdan17@gmail.com or +971545193040")

    contact_form = """
    <form action="https://formsubmit.co/snitdan17@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")





    # Title for the feedback page
    st.title("Feedback")

    # Add radio buttons for like and dislike options
    feedback = st.radio("Did you like this app?", ("Like", "Dislike"))

    # Display appropriate message based on the user's feedback
    if feedback == "Like":
        st.write("Thank you for your positive feedback!")
    else:
        st.write("We're sorry to hear that. Please let us know how we can improve.")


# Depending on the button selected in the sidebar, display the corresponding page
# if page == "authenticate":
#     authenticate_page()
if page == "Welcome Page" :
    welcome_page()
elif page == "Import":
    import_page()
elif page == "Display":
    display_page()
elif page == "Detection":
    detection_page()

elif page == "Contactus":
    contact_page()


import json

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_eSr9cajwxS.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",  # medium ; high
    height=None,
    width=None,
    key=None,
)
