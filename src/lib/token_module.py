import json
import logging

module_logger = logging.getLogger('icad_transcribe.token_module')


def add_token(db, token, token_name, ip_address, user_id):
    insert_query = """
    INSERT INTO api_tokens (token, token_name, token_ip_address ,user_id) VALUES (%s, %s, %s, %s)
    """
    params = (token, token_name, json.dumps(ip_address), user_id)
    return db.execute_commit(insert_query, params)

def get_tokens(db, token_id=None, token_name=None, user_id=None, token_value=None):
    base_query = """
    SELECT
      at.token_id,
      at.token,
      at.token_name,
      at.token_ip_address,
      at.user_id
    FROM api_tokens at
    """
    where_clauses = []
    parameters = []

    if token_id is not None:
        where_clauses.append("at.token_id = ?")
        parameters.append(token_id)
    if user_id is not None:
        where_clauses.append("at.user_id = ?")
        parameters.append(user_id)
    if token_name is not None:
        where_clauses.append("at.token_name = ?")
        parameters.append(token_name)

    if token_value is not None:
        where_clauses.append("at.token = ?")
        parameters.append(token_value)

    if where_clauses:
        where_clause = " WHERE " + " AND ".join(where_clauses)
        final_query = f"{base_query} {where_clause}"
    else:
        final_query = f"{base_query}"

    token_result = db.execute_query(final_query, params=parameters)

    # Convert any JSON-encoded IP address fields to Python lists
    for token in token_result.get('result', []):
        try:
            token['token_ip_address'] = json.loads(token.get('token_ip_address'))
        except (ValueError, TypeError):
            # If it doesn't parse, assume no valid IP list
            token['token_ip_address'] = []

    return token_result

def update_token(db, token_data):
    ip_address = token_data.get('token_ip_address')
    # Initialize IP address list
    ip_address_list = []

    # Add provided IP addresses if they exist
    if ip_address:
        if isinstance(ip_address, list):
            ip_address_list.extend(ip_address)
        elif isinstance(ip_address, str):
            # If given as a comma-separated string, split and add
            ip_address_list.extend([ip.strip() for ip in ip_address.split(",")])

    # Ensure there are no duplicate IPs and all entries are valid
    ip_address_list = list(set(ip_address_list))

    if len(ip_address_list) < 1:
        ip_address_list = ['*']

    update_query = f"""
    UPDATE
      api_tokens
    SET
       token = %s,
       token_name = %s,
       token_ip_address = %s
    WHERE
        token_id = %s
    """

    params = (token_data.get('token'),
              token_data.get('token_name'),
              json.dumps(ip_address_list),
              token_data.get('token_id'))

    update_result = db.execute_commit(update_query, params)

    return update_result

def delete_token(db, token_id):
    delete_query = "DELETE FROM api_tokens WHERE token_id = %s"
    params = (token_id,)
    delete_result = db.execute_commit(delete_query, params)
    return delete_result
