from agents.linkedin_lookup_agent import lookup

if __name__ == "__main__":
    linkednin_profile_url = lookup(name="Eden Marco")

    print(linkednin_profile_url)